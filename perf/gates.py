"""Shared production/smoke selection policy. Never select a failed dependency."""
def blocked(options,micro):
    errors=[]
    if micro.get('serialization',{}).get('original',{}).get('status')!='pass':errors.append('original_restore')
    if options.get('packed') and micro.get('serialization',{}).get('packed',{}).get('status')!='pass':errors.append('packed_restore')
    if options.get('direct_hash') and micro.get('hash_correctness',{}).get('status')!='pass':errors.append('direct_hash')
    if options.get('packed') and micro.get('json_correctness',{}).get('status')!='pass':errors.append('buffered_json')
    return errors

def choose(ranked):
    if not ranked:raise ValueError('NO_VALID_CANDIDATE')
    ordered=sorted(ranked,key=lambda x:x[0]);chosen=ordered[0][1]
    if chosen['label']=='screen-reference' and len(ordered)>1:chosen=ordered[1][1]
    return chosen

def dispatch_comparisons(run,valid,allowed):
    """One policy for formal comparisons and compressed smoke dispatch."""
    if not valid('candidate600') and allowed(650):run('candidate600',600,4801)
    if not valid('candidate600'):raise RuntimeError('CANDIDATE600_INCOMPLETE_OR_FAILED_AUDIT; no invalid comparison reported')
    if not valid('repeat-candidate300') and allowed(650):run('repeat-candidate300',300,4802)
    if not valid('repeat-baseline300') and allowed(340):run('repeat-baseline300',300,4802)
