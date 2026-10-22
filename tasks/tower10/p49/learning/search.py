"""P49 explicit evaluation budget; original P45 traversal and mean-Q backup."""
from tasks.tower10.p45.search import Search as OriginalSearch, Node


class Search(OriginalSearch):
    def begin(self, root, rng, evaluation=False):
        decision=super().begin(root,rng,evaluation=evaluation)
        if evaluation:
            decision['budget']=self.config['evaluation_new_simulations']
        if decision['budget'] != 100:raise ValueError('P49_SEARCH_BUDGET')
        return decision
