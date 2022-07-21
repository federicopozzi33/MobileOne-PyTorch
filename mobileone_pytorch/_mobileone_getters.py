from ._mobileone_network import MobileOneConfiguration, MobileOneNetwork, MobileOneSize, get_params


def mobileone_s0(in_channel: int = 3, num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S0, in_channel, num_classes=num_classes))


def mobileone_s1(in_channel: int = 3, num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S1, in_channel, num_classes=num_classes))


def mobileone_s2(in_channel: int = 3, num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S2, in_channel, num_classes=num_classes))


def mobileone_s3(in_channel: int = 3, num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S3, in_channel, num_classes=num_classes))


def mobileone_s4(in_channel: int = 3, num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S4, in_channel, num_classes=num_classes))


def _get_mobileone(conf: MobileOneConfiguration) -> MobileOneNetwork:
    return MobileOneNetwork(
        ks=conf.ks,
        out_channels=conf.out_channels,
        num_blocks=conf.num_blocks,
        strides=conf.strides,
        in_channel=conf.in_channel,
        num_classes=conf.num_classes,
    )
