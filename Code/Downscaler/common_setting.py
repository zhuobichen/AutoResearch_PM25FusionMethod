from dataclasses import dataclass


@dataclass
class CommonSetting:
    # V23: Conservative optimization based on v20
    # Default parameters: numit=2500, burn=500, thin=1
    numit: int = 2500
    burn: int = 500
    thin: int = 1
    neighbor: int = 3
    cmaqres: int = 12
    multithreading: bool = False
    modelfilepath: str | None = None

    def clone(self) -> "CommonSetting":
        return CommonSetting(
            numit=self.numit,
            burn=self.burn,
            thin=self.thin,
            neighbor=self.neighbor,
            cmaqres=self.cmaqres,
            multithreading=self.multithreading,
            modelfilepath=self.modelfilepath,
        )