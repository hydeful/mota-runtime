"""Original mean-Q PUCT and object actions, explicit 100 evaluation simulations."""
from tasks.tower10.p45.search import Search as OriginalSearch,Node


class Search(OriginalSearch):
    def begin(self,root,rng,evaluation=False):
        decision=super().begin(root,rng,evaluation=evaluation)
        if evaluation:decision['budget']=self.config['evaluation_new_simulations']
        if decision['budget']!=100:raise ValueError('TOPOLOGY_SEARCH_BUDGET')
        return decision
