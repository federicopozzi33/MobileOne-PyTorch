from ._mobileone_network import MobileOneConfiguration, MobileOneNetwork, MobileOneSize, get_params


def mobileone_s0(num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S0, num_classes))


def mobileone_s1(num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S1, num_classes=num_classes))


def mobileone_s2(num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S2, num_classes=num_classes))


def mobileone_s3(num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S3, num_classes=num_classes))


def mobileone_s4(num_classes: int = 1000) -> MobileOneNetwork:
    return _get_mobileone(get_params(MobileOneSize.S4, num_classes=num_classes))


def _get_mobileone(conf: MobileOneConfiguration) -> MobileOneNetwork:
    return MobileOneNetwork(
        ks=conf.ks,
        out_channels=conf.out_channels,
        num_blocks=conf.num_blocks,
        strides=conf.strides,
        num_classes=conf.num_classes,
    )
