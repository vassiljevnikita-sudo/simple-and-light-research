from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class SyntheticScenario:
    name: str; dates: tuple[int,...]; returns: dict[str,tuple[float,...]]
def known_truth_scenarios():
    d=tuple(range(60))
    return (SyntheticScenario('champion_decay',d,{'R01':tuple([.01]*30+[-.01]*30),'R02':tuple([0.0]*30+[.01]*30)}),
            SyntheticScenario('rare_profitable',d,{'R03':tuple(.02 if i%15==0 else 0 for i in d)}),
            SyntheticScenario('opportunity_response_failure',d,{'R07':tuple(.0 for _ in d)}),
            SyntheticScenario('recovery',d,{'R08':tuple([.01]*20+[-.01]*20+[.01]*20)}),
            SyntheticScenario('lucky_noise',d,{'R09':tuple([.03]*5+[0.0]*55)}),
            SyntheticScenario('null',d,{f'R{i:02d}':tuple(0.0 for _ in d) for i in range(1,11)}))
