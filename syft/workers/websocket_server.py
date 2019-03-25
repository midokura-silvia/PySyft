import binascii
import random
from typing import Union
from typing import List

#import asyncio
import torch
#import websockets
import syft as sy
from syft.codes import MSGTYPE
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.workers.virtual import VirtualWorker
from .websocket_client import WebsocketBridge, WebsocketClientWorker
from syft.workers import AbstractWorker
from syft.workers import IdProvider
from syft.codes import MSGTYPE
from typing import Union
from typing import List
from typing import Callable

from simple_websocket_server import WebSocketServer, WebSocket


class WebsocketServerWorker(VirtualWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        loop=None,
    ):
        """This is a simple extension to normal workers wherein
        all messages are passed over websockets. Note that because
        BaseWorker assumes a request/response paradigm, this worker
        enforces this paradigm by default.

        Args:
            hook (sy.TorchHook): a normal TorchHook object
            id (str or id): the unique id of the worker (string or int)
            log_msgs (bool): whether or not all messages should be
                saved locally for later inspection.
            verbose (bool): a verbose option - will print all messages
                sent/received to stdout
            host (str): the host on which the server should be run
            port (int): the port on which the server should be run
            data (dict): any initial tensors the server should be
                initialized with (such as datasets)
            loop: the asyncio event loop if you want to pass one in
                yourself
        """

        self.port = port
        self.host = host

        # call BaseWorker constructor
        super().__init__(hook=hook, id=id, data=data, log_msgs=log_msgs, verbose=verbose)

    async def _producer_handler(self, websocket):
        """This handler listens to the queue and processes messages as they
        arrive.

        Args:
            websocket: the connection object we use to send responses
                back to the client.

        """
        while True:

            # get a message from the queue
            message = await self.broadcast_queue.get()

            # convert that string message to the binary it represent
            message = binascii.unhexlify(message[2:-1])

            # process the message
            response = self.recv_msg(message)

            # convert the binary to a string representation
            # (this is needed for the websocket library)
            response = str(binascii.hexlify(response))

            # send the response
            await websocket.send(response)

    async def send_obj(self, tensor, worker):
        await worker.async_send_msg(MSGTYPE.OBJ, tensor)

    async def send(
        self,
        tensor: Union[torch.Tensor, AbstractTensor],
        worker: "BaseWorker",
        ptr_id: Union[str, int] = None,
    ) -> PointerTensor:
        """Sends tensor to the worker(s).
        Returns:
            A PointerTensor object representing the pointer to the remote worker(s).
        """

        if ptr_id is None:  # Define a remote id if not specified
            ptr_id = int(10e10 * random.random())

        pointer = tensor.create_pointer(
            owner=self, location=worker, id_at_location=tensor.id, register=True, ptr_id=ptr_id
        )

        # Send the object
        #await self.send_obj(tensor, worker)
        asyncio.run(self.send_obj(tensor, worker))
        raise "FOO"

        return pointer


    async def _handler(self, websocket):
        """Setup the consumer and producer response handlers with asyncio.

        Args:
            websocket: the websocket connection to the client

        """
#        tensor = a
#        ptr_id = int(10e10 * random.random())
#        pointer = tensor.create_pointer(
#            owner=self,
#            location=worker,
#            id_at_location=tensor.id,
#            register=True,
#            ptr_id=ptr_id
#        )
#        await worker.async_send_msg(MSGTYPE.OBJ, a, websocket)
        pointer = await self.send(a, worker)
        print(pointer)


    def handle(self):
        # echo message back to client
        self.send_message(self.data)

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')


    def start(self):
        """Start the server"""
        hook = self.hook

        class SocketClient(WebSocket):
            def handle(self):
                print("GOT FROM CLIENT", self.data)
                self.worker.messages.append(self.data)
#                self.worker.last_msg = self.data

            def connected(self):
                print('connected', self)

#                self.send_message("FOO")
                # plan
                a = torch.ones(2)
                a = a + 4
                print(a)
                self.worker = WebsocketBridge(hook=hook, client=self)
                a = a.send(self.worker)

        #            pointer = self.bridge.send(a, worker)



            def handle_close(self):
                print(self.address, 'closed')



        server = WebSocketServer(self.host, self.port, SocketClient)
        server.serve_forever()


