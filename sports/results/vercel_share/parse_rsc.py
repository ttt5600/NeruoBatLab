"""Decode Next.js RSC payload (self.__next_f.push([1,"..."])) from a saved Vercel leaderboard HTML -> one text blob.
Then extract JSON arrays of {"day":..,"metric":..,"chef_values":[...]} and similar day-keyed series."""
import re, json, sys
def rsc_text(html):
    out=[]
    for m in re.finditer(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html):
        out.append(json.loads('"'+m.group(1)+'"'))
    return "".join(out)
def day_objects(txt):
    """yield every JSON object that starts with {"day": by bracket matching"""
    for m in re.finditer(r'\{"day":"', txt):
        i=m.start(); depth=0; j=i; ins=False; esc=False
        while j < len(txt):
            c=txt[j]
            if ins:
                if esc: esc=False
                elif c=='\\': esc=True
                elif c=='"': ins=False
            else:
                if c=='"': ins=True
                elif c=='{': depth+=1
                elif c=='}':
                    depth-=1
                    if depth==0: break
            j+=1
        try: yield json.loads(txt[i:j+1])
        except Exception: pass
if __name__=="__main__":
    html=open(sys.argv[1]).read(); txt=rsc_text(html)
    print("rsc chars",len(txt))
    objs=list(day_objects(txt))
    keys={}
    for o in objs: keys.setdefault(tuple(sorted(o.keys())),0); keys[tuple(sorted(o.keys()))]+=1
    print(keys)
    if len(sys.argv)>2: open(sys.argv[2],"w").write(txt)
