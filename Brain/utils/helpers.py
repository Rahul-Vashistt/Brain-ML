from Brain.strategies import optimizers

def resolve_optimizer(name: str) -> str:
    name = name.lower()
    
    for key, aliases in optimizers.items():
        if name in [alias.lower() for alias in aliases]:
            return key
         
    raise ValueError(f"❌ Unknown optimizer '{name}'. Valid options are: {sum(optimizers.values(), [])}") 


def __lr_schedule(self,t: float) -> float:
        t0,t1 = 5,50
        return t0/(t+t1)