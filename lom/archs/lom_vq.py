"""
This script is adapted from the original implementation found at:
https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024

Author: Haiyang Liu
Modified by: Changan Chen, Juze Zhang and Shrinidhi Kowshika Lakshmikanth
License: Check the original repository for licensing details.
"""
import torch.nn as nn
from .tools.emage_quantizer import Quantizer
import torch



class VAEConv(nn.Module):
    def __init__(self, vae_layer=2, code_num=256, vae_test_dim=330, codebook_size=256, vae_quantizer_lambda=1.0):
        super(VAEConv, self).__init__()
        self.encoder = VQEncoderV3(vae_layer, code_num, vae_test_dim)
        self.decoder = VQDecoderV3(vae_layer, code_num, vae_test_dim)
        self.fc_mu = nn.Linear(code_num, code_num)
        self.fc_logvar = nn.Linear(code_num, code_num)
        self.reparameterize = reparameterize


    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        mu = self.fc_mu(pre_latent)
        logvar = self.fc_logvar(pre_latent)
        pre_latent =self.reparameterize(mu, logvar)
        rec_pose = self.decoder(pre_latent)
        return {
            "poses_feat": pre_latent,
            "rec_pose": rec_pose,
            "pose_mu": mu,
            "pose_logvar": logvar,
        }

class VQEncoderV3(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQEncoderV3, self).__init__()
        n_down = vae_layer
        channels = [code_num]
        for i in range(n_down - 1):
            channels.append(code_num)

        input_size = vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQDecoderV3(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQDecoderV3, self).__init__()
        n_up = vae_layer
        channels = []
        for i in range(n_up - 1):
            channels.append(code_num)
        channels.append(code_num)
        channels.append(vae_test_dim)
        input_size = code_num
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

# VQVAE Class
class VQVAEConvZeroDSUS(nn.Module):
    def __init__(self, vae_layer=2, code_num=256, vae_test_dim=300, codebook_size=256, vae_quantizer_lambda=1.0):
        super().__init__()
        self.encoder = VQEncoderV5DS(vae_layer, code_num, vae_test_dim)
        self.quantizer = Quantizer(codebook_size, code_num, vae_quantizer_lambda)
        self.decoder = VQDecoderV5US(vae_layer, code_num, vae_test_dim)

    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat": vq_latent,
            "embedding_loss": embedding_loss,
            "perplexity": perplexity,
            "rec_pose": rec_pose
        }

    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index

    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q

    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose




class VQEncoderDS(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQEncoderDS, self).__init__()
        n_down = vae_layer
        channels = [vae_test_dim] + ([code_num] * n_down)
        layers = list()
        for i in range(n_down):
            stride = 1 if i == 0 else 2
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 3, stride, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(ResBlock(channels[i + 1]))
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class VQDecoderUS(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQDecoderUS, self).__init__()
        n_up = vae_layer
        channels = code_num
        out_dim = vae_test_dim
        layers = list()
        for i in range(n_up):
            layers.append(ResBlock(channels))
            if i < n_up - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv1d(channels, channels, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv1d(channels, out_dim, 3, 1, 1))
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class VQVAEConvZero(nn.Module):
    def __init__(self, vae_layer=2, code_num=256, vae_test_dim=300, codebook_size=256, vae_quantizer_lambda=1.0):
        super(VQVAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(vae_layer, code_num, vae_test_dim)
        self.quantizer = Quantizer(codebook_size, code_num, vae_quantizer_lambda)
        self.decoder = VQDecoderV5(vae_layer, code_num, vae_test_dim)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat":vq_latent,
            "embedding_loss":embedding_loss,
            "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    

# VQVAE Class
class VQVAEConvZeroDSUS_PaperVersion(nn.Module):
    def __init__(self, vae_layer=2, code_num=256, vae_test_dim=300, codebook_size=256, vae_quantizer_lambda=1.0):
        super().__init__()
        self.encoder = VQEncoderDS(vae_layer, code_num, vae_test_dim)
        self.quantizer = Quantizer(codebook_size, code_num, vae_quantizer_lambda)
        self.decoder = VQDecoderUS(vae_layer, code_num, vae_test_dim)

    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {
            "poses_feat": vq_latent,
            "embedding_loss": embedding_loss,
            "perplexity": perplexity,
            "rec_pose": rec_pose
        }

    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index

    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q

    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose



# VQVAE Encoder
class VQEncoderV5DS(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super().__init__()
        n_down = vae_layer
        channels = [code_num]
        for i in range(n_down - 1):
            channels.append(code_num)

        input_size = vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]
        ## down sampling
        if vae_layer == 1:
            layers += [
                nn.Conv1d(channels[0], channels[0], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[0]),
            ]
        elif vae_layer == 2:
            layers += [
                nn.Conv1d(channels[1], channels[1], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[1]),
            ]
        ## keep deq dim
        # if vae_layer > 2:
        #     for i in range(2, vae_layer - 2):
        #         layers += [
        #             nn.Conv1d(channels[i - 1], channels[i], 3, 1, 1),
        #             nn.LeakyReLU(0.2, inplace=True),
        #             ResBlock(channels[i]),
        #         ]

        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


# VQVAE Decoder
class VQDecoderV5US(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super().__init__()
        # vae_layer = 2  # args.vae_layer
        channels = []
        for i in range(vae_layer):
            channels.append(code_num)
        # channels.append(code_num)
        channels.append(vae_test_dim)
        input_size = code_num
        n_resblk = 2
        assert len(channels) == vae_layer + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(vae_layer):
            up_factor = 2  # 2 if i < n_up - 1 else 1
            layers += [
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # if vae_layer > 2:
        #     for i in range(2, vae_layer):
        #         layers += [
        #             nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),
        #             nn.LeakyReLU(0.2, inplace=True)
        #         ]

        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std




class VAEConvZero(nn.Module):
    def __init__(self, vae_layer=4, code_num=256, vae_test_dim=61, codebook_size=256, vae_quantizer_lambda=1.0):
        super(VAEConvZero, self).__init__()
        self.encoder = VQEncoderV5(vae_layer, code_num, vae_test_dim)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(vae_layer, code_num, vae_test_dim)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        # print(pre_latent.shape)
        # embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(pre_latent)
        return {
            # "poses_feat":vq_latent,
            # "embedding_loss":embedding_loss,
            # "perplexity":perplexity,
            "rec_pose": rec_pose
            }
    



class VQEncoderV5(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQEncoderV5, self).__init__()
        n_down = vae_layer
        channels = [code_num]
        for i in range(n_down-1):
            channels.append(code_num)
        
        input_size = vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


class VQDecoderV5(nn.Module):
    def __init__(self, vae_layer, code_num, vae_test_dim):
        super(VQDecoderV5, self).__init__()
        n_up = vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(code_num)
        channels.append(code_num)
        channels.append(vae_test_dim)
        input_size = code_num
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            up_factor = 2 if i < n_up - 1 else 1
            layers += [
                #nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
