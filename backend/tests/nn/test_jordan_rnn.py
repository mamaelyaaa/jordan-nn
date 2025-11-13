import pytest

from network.training.activation import SigmoidActivation, LinearActivation
from network.jordan import JordanRNN
from network.training.layers import HiddenLayer, OutputLayer


@pytest.fixture
def samples():

    return


def test_jordan_rnn():
    jnn = JordanRNN(
        hidden_layer=HiddenLayer(neurons=5, activation=SigmoidActivation()),
        output_layer=OutputLayer(neurons=1, activation=LinearActivation()),
        learning_rate=0.01,
    )

    jnn.train(training=..., targets=..., epochs=100)
