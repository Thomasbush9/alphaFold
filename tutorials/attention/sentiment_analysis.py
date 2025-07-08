import torch
import torch.nn as nn
import pytorch_lightning as pl
from attention.mha import MultiHeadAttention


class AttentionBlock(nn.Module):
    """
    Implements an AttentionBlock following the default architecture:
    MultiHeadAttention - Add & Norm - Intermediate - Add & Norm
    """

    def __init__(self, hidden_size, intermediate_size, N_head):
        """
        Initialize AttentionBlock for a transformer.

        Args:
            hidden_size (int): Dimension of the module's input and output.
            intermediate_size (int): Hidden dimension for the intermediate
            feed-forward module.
            N_head (int): Number of heads for multi-head attention.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules mha (MultiHeadAttention),                 #
        #   layer_norm_1 (LayerNorm), intermediate (Sequential conisting of      #
        #   Linear, GELU, Linear) and layer_norm_2 (LayerNorm).                  #
        #   The query, key and value embeddings have dimension                   #
        #   hidden_size/N_head, so that they are of size hidden_size again       #
        #   after concatenation.                                                 #
        ##########################################################################

        # Replace "pass" statement with your code
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.N_head = N_head
        self.c = hidden_size // N_head


        self.mha = MultiHeadAttention(c_in=self.hidden_size, c=self.c, attn_dim=-2,  N_head=self.N_head, use_bias_for_embeddings=True)
        self.layer_norm_1 = nn.LayerNorm(self.hidden_size)
        self.intermediate = nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size),
                nn.GELU(),
                nn.Linear(self.intermediate_size, self.hidden_size),)
        self.layer_norm_2 = nn.LayerNorm(self.hidden_size)


        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x, attention_mask=None):
        """
        Forward pass for AttentionBlock. The forward pass consists of the steps
        x -> mha -> h1 -> layer_norm(x+h1) -> intermediate -> h2 -> layer_norm(x+h2)

        Args:
            x (torch.tensor): Input tensor of shape (*, T, c) where T denotes the
                temporal dimension along which attention is performed, and c is the hidden_size.
            attention_mask (torch.tensor, optional): Attention mask of
                shape (*, k). If not None, values that are equal to zero
                in the mask are masked out during attention. Defaults to None.

        Returns:
            torch.tensor: Output tensor of shape (*, T, c).
        """

        out = None
        ##########################################################################
        # TODO: Compute the forward pass for the layer as described above.       #
        ##########################################################################

        # Replace "pass" statement with your code

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        print(x.shape)
        h1 = self.mha(x, attention_mask=attention_mask)
        x1 = self.layer_norm_1(x + h1)         # after attention + residual + norm

        h2 = self.intermediate(x1)            # feedforward
        out = self.layer_norm_2(x1 + h2)
        return out


class SentimentAnalysis(nn.Module):
    """
    Implements a transformer encoder for the use of sentiment analysis.
    The module uses learned positional encodings. After the attention
    blocks, the output at the position of the <Start> token is fed
    through a two-layer feed-forward classifier to classify the input
    as either negative or positive.

    Attributes:
        vocab_size (int): The number of tokens in the vocabulary.
        hidden_size (int): Dimension of input and output in attention blocks,
            as well as word and position embeddings.
        intermediate_size (int): Dimension of the intermediate layer in the
            feed-forward module of the attention blocks.
        N_head (int): Number of heads in the attention blocks.
        num_blocks (int): Number of attention blocks.
        input_length (int): Number of tokens in the module input. All input
            must be cropped or padded to this length.
    """

    def __init__(self, vocab_size, hidden_size, intermediate_size, N_head, num_blocks, input_length):
        """
        Initialize the SentimentAnalysis module.

        Args:
            vocab_size (int): The number of tokens in the vocabulary.
            hidden_size (int): Dimensions of input and output in attention blocks,
                as well as word and position embeddings. hidden_size should be
                divisible by N_head.
            intermediate_size (int): Dimension of the intermediate layer in the
                feed-forward module of the attention blocks.
            N_head (int): Number of heads in the attention blocks.
            num_blocks (int): Number of attention blocks.
            input_length (int): Number of tokens in the module input. All input
                must be cropped or padded to this length.
        """
        super().__init__()
        self.input_length = input_length

        ##########################################################################
        # TODO: Initialize the modules word_embeddings, position_embeddings,     #
        #   layer_norm, blocks (ModuleList of num_blocks attention blocks) and   #
        #   out (linear - ReLU - linear sequential module).                      #
        #   For the word and position embeddings, use the module nn.Embedding.   #
        #   For the word embeddings, set the argument padding_idx                #
        #   to zero (the padding token in the vocabulary).                       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, inp, attention_mask=None):
        """
        Forward pass for the SentimentAnalysis module consisting of the following steps:

        1. word_embeddings + position_embeddings
        2. Layer normalization
        3. Multiple attention blocks
        4. Extract the representation at the <Start> token position (at index 0).
        5. Feed-forward

        Args:
            inp (torch.tensor): Input tensor of shape (*, T) consisting of
                the token indices in the input sentence. Here, T is the input length.
            attention_mask (torch.tensor, optional): Attention mask of
                shape (*, k). If not None, values that are equal to zero
                in the mask are masked out in all attention blocks. Defaults to None.

        Returns:
            torch.tensor: Output tensor of shape (*, 2).
        """

        out = None

        ##########################################################################
        # TODO: Compute the forward pass as described above. You can construct   #
        #   the indices to select the position embeddings using `torch.arange`.  #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class SentimentWrapper(pl.LightningModule):

    def __init__(self, model, learning_rate=2e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        ##########################################################################
        # TODO: Initialize criterion as `CrossEntropyLoss`.                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


    def forward(self, inp, attention_mask=None):
        out = None

        ##########################################################################
        # TODO: Forward the input to the wrapped SentimentAnalysis module.       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

    def training_step(self, batch, batch_idx):
        inp, attn_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']

        out, loss, accuracy = None, None, None

        ##########################################################################
        # TODO: Compute the forward pass, the loss (using the criterion you      #
        #   initialized in __init__) and the accuracy on the batch.              #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        ##########################################################################
        # TODO: Destructure the batch, compute out, loss and accuracy, and log   #
        #   the  values as val_loss and val_acc as in training_step.             #
        #   Validation only happens every epoch anyway, so you don't need to     #
        #   specify on_epoch for logging.                                        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def configure_optimizers(self):
        optimizer = None

        ##########################################################################
        # TODO: Set optimizer to a torch.optim.AdamW optimizer. Set the          #
        #   parameters to this model's parameters and the learning rate          #
        #   self.learning_rate.                                                  #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return optimizer


def map_keynames_from_distilbert(named_parameters):
    name_map = {
        'distilbert.': '',
        'embeddings.LayerNorm': 'layer_norm',
        'embeddings.position_embeddings': 'position_embeddings',
        'embeddings.word_embeddings': 'word_embeddings',
        'transformer.layer.': 'blocks.',
        'attention.': 'mha.',
        'q_lin.': 'linear_q.',
        'k_lin.': 'linear_k.',
        'v_lin.': 'linear_v.',
        'out_lin.': 'linear_o.',
        'sa_layer_norm': 'layer_norm_1',
        'ffn.lin1': 'intermediate.0',
        'ffn.lin2': 'intermediate.2',
        'output_layer_norm': 'layer_norm_2',
        'pre_classifier': 'out.0',
        'classifier': 'out.2',

    }

    new_parameters = dict()
    if isinstance(named_parameters, dict):
        named_parameters = named_parameters.items()

    for i, (key, value) in enumerate(named_parameters):
        for original, new in name_map.items():
            key = key.replace(original, new)
        new_parameters[key] = value

    return new_parameters
