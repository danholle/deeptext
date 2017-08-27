#!/usr/bin/python3

import json

def tryit():
  with open("props.json","r") as f:
    props=json.load(f)
    f.close()
  tuples=[(1,1.09,3,7.2)]
  tuples.append((2,2.1,4,8.9))
  tuples.append((3,3.14,5,7.1))
  props["valprogress"]=tuples

  with open("propso.json","w") as f:
    json.dump(props,f,indent=2,sort_keys=True)
    f.close()
# end def tryit

if __name__ == '__main__':
  tryit()


