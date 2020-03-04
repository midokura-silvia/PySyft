import pytest

from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy

from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker
from syft.frameworks.torch.fl import utils

PRINT_IN_UNITTESTS = True


# TODO: I'm not sure this is valid torch JIT anyway
@pytest.mark.skip(reason="fails currently as it needs functions as torch.nn.linear to be unhooked.")
def test_train_config_with_jit_script_module(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real.float() - pred.float()) ** 2).mean()

    # Model
    class Net(torch.jit.ScriptModule):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @torch.jit.script_method
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Net()
    model.id = sy.ID_PROVIDER.pop()

    loss_fn.id = sy.ID_PROVIDER.pop()

    model_ptr = me.send(model, alice)
    loss_fn_ptr = me.send(loss_fn, alice)

    # Create and send train config
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset="vectors")
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    if PRINT_IN_UNITTESTS:
        print(alice)
    new_model = model_ptr.get()
    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)

    pred = new_model(data)
    loss_after = loss_fn(real=target, pred=pred)

    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))

    assert loss_after < loss_before


def test_train_config_with_jit_trace(hook, workers):  # pragma: no cover
    alice = workers["alice"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="gaussian_mixture")

    @torch.jit.script
    def loss_fn(pred, target):
        return ((target.float().view_as(pred) - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.bn = nn.BatchNorm1d(num_features=2)
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        def forward(self, x):
            x = self.bn(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, data)

    if PRINT_IN_UNITTESTS:
        print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)

    if PRINT_IN_UNITTESTS:
        print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=64)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key="gaussian_mixture")
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)

    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))
        print_training_result(pred, target)

    assert loss_after < loss_before


def test_train_config_with_jit_trace_send_twice_with_fit(hook, workers):  # pragma: no cover
    torch.manual_seed(0)
    alice = workers["alice"]
    model, loss_fn, data, target, loss_before, dataset_key = prepare_training(
        hook, alice, nr_samples=100, mu_0=-10, mu_1=10
    )

    # Create and send train config
    trainconfig_args = {
        "model": model,
        "loss_fn": loss_fn,
        "batch_size": 16,
        "epochs": 100,
        "optimizer_args": dict({"lr": 0.01, "weight_decay": 0.9}),
    }
    train_config_0 = sy.TrainConfig(**trainconfig_args)
    train_config_0.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("TrainConfig 0, iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config_0.model_ptr.get()
    pred = new_model.obj(data)
    loss_after_0 = loss_fn(pred=pred, target=target)

    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss after training with TrainConfig 0: {}".format(loss_after_0))
        print_training_result(pred, target)

    assert loss_after_0 < loss_before

    train_config = sy.TrainConfig(**trainconfig_args)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key=dataset_key)

        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("TrainConfig 1, iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(pred=pred, target=target)
    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss after training with TrainConfig 0: {}".format(loss_after_0))
        print("Loss after training with TrainConfig 1: {}".format(loss_after))
        print_training_result(pred, target)
        print_trained_model_weights_and_bias(model)

    local_training_with_train_config(
        model=model, train_config=train_config, data=data, target=target, loss_fn=loss_fn
    )

    assert loss_after < loss_before


def print_training_result(pred, target):  # pragma: no cover
    print("Predictions: {}".format(pred.detach().numpy().reshape(1, -1)))
    print("Targets    : {}".format(target.reshape(1, -1)))
    print(
        "Accuracy: {}".format(
            ((pred.view_as(target) - target).abs() < 0.5).sum() / float(len(target))
        )
    )


def print_trained_model_weights_and_bias(model):  # pragma: no cover
    print("fc1.weight: {}".format(model.fc1._parameters["weight"]))
    print("fc2.weight: {}".format(model.fc2._parameters["weight"]))
    print("fc3.weight: {}".format(model.fc3._parameters["weight"]))
    print("fc1.bias: {}".format(model.fc1._parameters["bias"]))
    print("fc2.bias: {}".format(model.fc2._parameters["bias"]))
    print("fc3.bias: {}".format(model.fc3._parameters["bias"]))


def local_training_with_train_config(model, loss_fn, train_config, data, target):
    model.train()
    optimizer_class = getattr(torch.optim, train_config.optimizer)
    optimizer = optimizer_class(model.parameters(), **train_config.optimizer_args)
    # optimizer = torch.optim.Adam(model.parameters(), **(train_config.optimizer_args))
    for epoch in range(train_config.epochs * 5):
        output = model(data)
        loss = loss_fn(pred=output, target=target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss))


def prepare_training(alice, nr_samples=100, mu_0=-1, mu_1=1):  # pragma: no cover
    model, loss_fn, data, target, loss_before, dataset_key, dataset, model_untraced = prepare_training_additional_output(
        alice, nr_samples=nr_samples, mu_0=mu_0, mu_1=mu_1
    )
    return model, loss_fn, data, target, loss_before, dataset_key


def prepare_training_additional_output(alice, nr_samples=100, mu_0=-1, mu_1=1):  # pragma: no cover
    data, target = utils.create_gaussian_mixture_toy_data(
        nr_samples=nr_samples, mu_0=mu_0, mu_1=mu_1
    )
    dataset_key = "gaussian_mixture"

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key=dataset_key)

    @torch.jit.script
    def loss_fn(pred, target):
        return ((target.float() - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # self.bn = nn.BatchNorm1d(num_features=2)
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            # x = self.bn(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, data)

    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)
    return model, loss_fn, data, target, loss_before, dataset_key, dataset, model_untraced


def test___str__():
    train_config = sy.TrainConfig(batch_size=2, id=99887766, model=None, loss_fn=None)

    train_config_str = str(train_config)
    str_expected = (
        "<TrainConfig id:99887766 owner:me epochs: 1 batch_size: 2 optimizer_args: {'lr': 0.1}>"
    )

    assert str_expected == train_config_str


def test_local_training(workers):
    alice = workers["alice"]

    mu_0 = 1
    mu_1 = 5
    model, loss_fn, data, target, loss_before, dataset_key, dataset, model_untraced = prepare_training_additional_output(
        alice, nr_samples=64, mu_0=mu_0, mu_1=mu_1
    )
    print_trained_model_weights_and_bias(model_untraced)
    train_config = sy.TrainConfig(
        model=None,
        loss_fn=None,
        batch_size=10,
        optimizer="SGD",
        optimizer_args={"lr": 0.001, "weight_decay": 0.01},
        epochs=10,
        shuffle=True,
    )
    local_training_with_train_config(
        model=model_untraced, loss_fn=loss_fn, train_config=train_config, data=data, target=target
    )

    print_trained_model_weights_and_bias(model_untraced)

    data_test, target_test = utils.create_gaussian_mixture_toy_data(
        nr_samples=64, mu_0=mu_0, mu_1=mu_1
    )


def test_send(workers):
    alice = workers["alice"]

    train_config = sy.TrainConfig(batch_size=2, id="id", model=None, loss_fn=None)
    train_config.send(alice)

    assert alice.train_config.id == train_config.id
    assert alice.train_config._model_id == train_config._model_id
    assert alice.train_config._loss_fn_id == train_config._loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.optimizer_args == train_config.optimizer_args
    assert alice.train_config.location == train_config.location


def test_send_model_and_loss_fn(workers):
    train_config = sy.TrainConfig(
        batch_size=2, id="send_model_and_loss_fn_tc", model=None, loss_fn=None
    )
    alice = workers["alice"]

    orig_func = sy.ID_PROVIDER.pop
    model_id = 44
    model_id_at_location = 44000
    loss_fn_id = 55
    loss_fn_id_at_location = 55000
    sy.ID_PROVIDER.pop = mock.Mock(
        side_effect=[model_id, model_id_at_location, loss_fn_id, loss_fn_id_at_location]
    )

    train_config.send(alice)

    assert alice.train_config.id == train_config.id
    assert alice.train_config._model_id == train_config._model_id
    assert alice.train_config._loss_fn_id == train_config._loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.optimizer_args == train_config.optimizer_args
    assert alice.train_config.location == train_config.location
    assert alice.train_config._model_id == model_id
    assert alice.train_config._loss_fn_id == loss_fn_id

    sy.ID_PROVIDER.pop = orig_func


@pytest.mark.asyncio
async def test_train_config_with_jit_trace_async(hook, start_proc):  # pragma: no cover
    kwargs = {"id": "async_fit", "host": "localhost", "port": 8777, "hook": hook}
    data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)
    dataset_key = "gaussian_mixture"

    mock_data = torch.zeros(1, 2)

    # TODO check reason for error (RuntimeError: This event loop is already running) when starting websocket server from pytest-asyncio environment
    # dataset = sy.BaseDataset(data, target)

    # server, remote_proxy = start_remote_worker(id="async_fit", port=8777, hook=hook, dataset=(dataset, dataset_key))

    # time.sleep(0.1)

    remote_proxy = WebsocketClientWorker(**kwargs)

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # self.bn = nn.BatchNorm1d(num_features=1)
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        def forward(self, x):
            # x = self.bn(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, mock_data)

    pred_before = model(data)
    loss_before = loss_fn(target=target, pred=pred_before)

    # Create and send train config
    train_config = sy.TrainConfig(
        model=model,
        loss_fn=loss_fn,
        batch_size=2,
        optimizer="SGD",
        optimizer_args={"lr": 0.1},
        epochs=2,
        shuffle=True,
    )
    train_config.send(remote_proxy)

    for epoch in range(5):
        loss = await remote_proxy.async_fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()

    assert not (model.fc1._parameters["weight"] == new_model.obj.fc1._parameters["weight"]).all()
    assert not (model.fc2._parameters["weight"] == new_model.obj.fc2._parameters["weight"]).all()
    assert not (model.fc3._parameters["weight"] == new_model.obj.fc3._parameters["weight"]).all()
    assert not (model.fc1._parameters["bias"] == new_model.obj.fc1._parameters["bias"]).all()
    assert not (model.fc2._parameters["bias"] == new_model.obj.fc2._parameters["bias"]).all()
    assert not (model.fc3._parameters["bias"] == new_model.obj.fc3._parameters["bias"]).all()

    new_model.obj.eval()
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Loss before training: {}".format(loss_before))
        print_training_result(pred_before, target)
        print("Loss after training: {}".format(loss_after))
        print_training_result(pred, target)

    remote_proxy.close()
    # server.terminate()

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print_trained_model_weights_and_bias(model)

    assert loss_after < loss_before


def test_train_config_with_jit_trace_sync(hook, start_remote_worker):  # pragma: no cover
    data, target = utils.create_gaussian_mixture_toy_data(100)
    dataset = sy.BaseDataset(data, target)
    dataset_key = "gaussian_mixture"

    server, remote_proxy = start_remote_worker(
        id="sync_fit", hook=hook, port=9000, dataset=(dataset, dataset_key)
    )

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, data)

    pred = model(data)
    loss_before = loss_fn(pred=pred, target=target)

    # Create and send train config
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2, epochs=1)
    train_config.send(remote_proxy)

    for epoch in range(5):
        loss = remote_proxy.fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()

    # assert that the new model has updated (modified) parameters
    assert not (
        (model.fc1._parameters["weight"] - new_model.obj.fc1._parameters["weight"]).abs() < 10e-3
    ).all()
    assert not (
        (model.fc2._parameters["weight"] - new_model.obj.fc2._parameters["weight"]).abs() < 10e-3
    ).all()
    assert not (
        (model.fc3._parameters["weight"] - new_model.obj.fc3._parameters["weight"]).abs() < 10e-3
    ).all()
    assert not (
        (model.fc1._parameters["bias"] - new_model.obj.fc1._parameters["bias"]).abs() < 10e-3
    ).all()
    assert not (
        (model.fc2._parameters["bias"] - new_model.obj.fc2._parameters["bias"]).abs() < 10e-3
    ).all()
    assert not (
        (model.fc3._parameters["bias"] - new_model.obj.fc3._parameters["bias"]).abs() < 10e-3
    ).all()

    new_model.obj.eval()
    pred = new_model.obj(data)
    loss_after = loss_fn(pred=pred, target=target)

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))
        print_training_result(pred, target)

    remote_proxy.close()
    server.terminate()

    assert loss_after < loss_before
