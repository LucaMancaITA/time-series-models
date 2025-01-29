import torch
import torch.nn as nn

from ts_forecasting.torch_app.net.utils.embed import DataEmbedding_wo_pos
from ts_forecasting.torch_app.net.utils.corr import AutoCorrelation, AutoCorrelationLayer
from ts_forecasting.torch_app.net.utils.encoder import EncoderDecomp, EncoderLayerDecomp
from ts_forecasting.torch_app.net.utils.decoder import DecoderDecomp, DecoderLayerDecomp
from ts_forecasting.torch_app.net.utils.decomposition import series_decomp
from ts_forecasting.torch_app.net.utils.act import my_Layernorm


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, moving_avg=1, dropout=0.0, embed='fixed',
                 freq='h', activation='gelu', output_attention=False):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            enc_in, d_model, embed, freq,
            dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            dec_in, d_model, embed, freq, dropout
        )

        # Encoder
        self.encoder = EncoderDecomp(
            [
                EncoderLayerDecomp(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor,
                                        attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = DecoderDecomp(
            [
                DecoderLayerDecomp(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor,
                                        attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor,
                                        attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]],
            device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask,
            cross_mask=dec_enc_mask, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == "__main__":

    seq_len = 96
    pred_len = 24
    label_len = 24
    batch_x = torch.rand(size=(32, seq_len, 1))
    batch_x_mark = torch.rand(size=(32, seq_len, 4))
    batch_y = torch.rand(size=(32, pred_len+label_len, 1))
    batch_y_mark = torch.rand(size=(32, pred_len+label_len, 4))
    dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:,:label_len,:], dec_inp], dim=1).float()

    model = Autoformer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        moving_avg=1,
        dropout=0.0,
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False
    )

    print(batch_x.size(), batch_x_mark.size(), dec_inp.size(), batch_y_mark.size())
    y = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print(y.size())