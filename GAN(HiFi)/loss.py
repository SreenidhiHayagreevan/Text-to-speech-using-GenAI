import torch

def feature_loss(feature_map_real, feature_map_gen, lambda_features=2):
    loss = 0
    for features_real, features_gen in zip(feature_map_real, feature_map_gen):
        for feature_real, feature_gen in zip(features_real, features_gen):
            loss += torch.mean(torch.abs(feature_real - feature_gen))

    return loss*lambda_features


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l
    return loss

