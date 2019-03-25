import binascii
import random
import asyncio
from typing import Union
from typing import List
import syft as sy
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.frameworks.torch.tensors.interpreters import AbstractTensor

import torch
import websocket
import time
from websocket import create_connection
import threading

from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers import BaseWorker


class WebsocketBridge(BaseWorker):
    def __init__(
        self,
        hook,
        client,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """
        self.client = client
        self.last_msg = None
        self.messages = []

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def x_send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )

    def _recv_msg(self, message: bin):
        """Forwards a message to the WebsocketServerWorker"""
        print("RCV", message)
        hmessage = str(binascii.hexlify(message))
        self.client.send_message(hmessage)

        print(self.client.sendq)
        print("handle")
        self.client._handle_data()
        print("pop")
        print(self.client.sendq)



        response = binascii.unhexlify(self.last_msg[2:-1])
        return message


    def _send_msg(self, bin_message):
        print('sending', bin_message)
        bin_message = str(binascii.hexlify(bin_message))
        self.client.send_message(bin_message)
        return binascii.unhexlify(resp[2:-1])


class WebsocketClientWorker(BaseWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        # TODO get angry when we have no connection params
        self.port = port
        self.host = host
        self.uri = f"ws://{self.host}:{self.port}"
        self.verbose = True

        # creates the connection with the server which gets held open until the
        # WebsocketClientWorker is garbage collected.
#        self.ws = create_connection(self.uri)

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )


    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketServerWorker"""
        return self.recv_msg(message)

#        self.ws.send(str(binascii.hexlify(message)))
#        response = binascii.unhexlify(self.ws.recv()[2:-1])
#        return response


    def ready_to_compute(self):
        def on_message(ws, message):
            print("GOT", message)
            b = binascii.unhexlify(message[2:-1])
            response = self.recv_msg(b)
            response = str(binascii.hexlify(response))
            print('sending' , response)
            ws.send(response)

        def on_error(ws, error):
            print(error)

        def on_close(ws):
            print("### closed ###")

#        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(self.uri,
                    on_message = on_message,
                    on_error = on_error,
                    on_close = on_close)
        print('listening')
        self.ws.run_forever()
