def integer(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def percent(share: float, digits: int = 1) -> str:
    return f"{share * 100:.{digits}f} %".replace(".", ",")


def signed_percent(share: float, digits: int = 1) -> str:
    return f"{share * 100:+.{digits}f} %".replace(".", ",")


def decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")
