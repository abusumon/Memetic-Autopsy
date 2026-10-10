import csv,subprocess,sys,time,os
budget=float(sys.argv[1]); t0=time.time()
jobs=[tuple(j.split('|')) for j in open('queue.txt').read().split('\n') if j.strip() and not j.startswith('#')]
# job: out|instance|cond|runs|gens|theta|repl|tag
def done(out,inst,cond):
    if not os.path.exists(out): return 0
    return sum(1 for r in csv.DictReader(open(out)) if r['instance']==inst and r['condition']==cond)
for out,inst,cond,runs,gens,theta,repl,tag in jobs:
    name=cond+tag; have=done(out,inst,name); need=int(runs)-have
    if need<=0: continue
    left=budget-(time.time()-t0)
    if left<25: break
    print(f"JOB {inst} {name}: have {have}/{runs}",flush=True)
    try:
        subprocess.run(['./driver','--datasets',inst,'--conds',cond,'--runs',str(need),'--gens',gens,'--theta',theta,'--repl',repl,'--tag',tag,'--out',out,'--seed0',str(have+1)],timeout=left,stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print("  (time budget hit; will resume)",flush=True); break
print("elapsed",round(time.time()-t0))
