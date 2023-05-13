import os, sys, json

class SafeTensorsException(Exception):
    """Exception raised when something wrong with the file.

    Attributes:
        filename -- path to .safetensors file
        whatiswrong -- string describing what's wrong
    """
    def __init__(self, filename:str, whatiswrong:str):
        self.filename = filename
        self.whatiswrong = whatiswrong
        super().__init__()

    def __str__(self):
        return f"{self.filename} is not a valid .safetensors file: {self.whatiswrong}"

class SafeTensorsFile:
    def __init__(self):
        self.f=None
        self.hdrbuf=None

    def __del__(self):
        self.close()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f=None

    def open(self,fn:str,openMode='rb',quiet=False)->int:
        try:
            st=os.stat(fn)
            if st.st_size<8:
                raise SafeTensorsException(fn,"length less than 8 bytes")

            f=open(fn,openMode)
            b8=f.read(8) #read header size
            if len(b8)!=8:
                raise SafeTensorsException(fn,f"read only {len(b8)} bytes at start of file")
            headerlen=int.from_bytes(b8,'little',signed=False)
            if headerlen==0:
                raise SafeTensorsException(fn,"header size is 0")
            if (8+headerlen>st.st_size): 
                raise SafeTensorsException(fn,"header extends past end of file")

            if quiet==False:
                print(f"{fn}: length={st.st_size}, header length={headerlen}")
            hdrbuf=f.read(headerlen)
            if len(hdrbuf)!=headerlen:
                raise SafeTensorsException(fn,f"header size is {headerlen}, but read {len(hdrbuf)} bytes")
        except SafeTensorsException as e:
            print(e,file=sys.stderr)
            return -1

        self.fn=fn
        self.f=f
        self.st=st
        self.hdrbuf=hdrbuf
        return 0

    def parse_buffer(self):
        if self.hdrbuf is None: return None
        try:
            return json.loads(self.hdrbuf)
        except json.JSONDecodeError as e:
            print("json.JSONDecodeError when parsing file header:",file=sys.stderr)
            print(e,file=sys.stderr)
            return None


def PrintHeader(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile()
    s.open(input_file)
    js=s.parse_buffer()
    if js is None: return -1

    # All the .safetensors files I've seen have long key names, and as a result, 
    # the neither json nor pprint package prints text in very readable format,
    # so we print it ourselves, putting key name & value on one long line.
    # Note the print out is in Python format, not valid JSON format.
    firstKey=True
    print("{")
    for key in js:
        if firstKey:
            firstKey=False
        else:
            print(",")
        json.dump(key,fp=sys.stdout,ensure_ascii=False,separators=(',',':'))
        print(": ",end='')
        json.dump(js[key],fp=sys.stdout,ensure_ascii=False,separators=(',',':'))
    print("\n}")
    return 0
    
def _ParseMore(d:dict):
    '''Basically when printing, try to turn this:

        "ss_dataset_dirs":"{\"abc\": {\"n_repeats\": 2, \"img_count\": 60}}",

    into this:

        "ss_dataset_dirs":{
         "abc":{
          "n_repeats":2,
          "img_count":60
         }
        },

    '''
    for key in d:
        value=d[key]
        #print("+++",key,value,type(value),"+++",sep='|')
        if isinstance(value,str):
            try:
                v2=json.loads(value)
                d[key]=v2
                value=v2
            except json.JSONDecodeError as e:
                pass    
        if isinstance(value,dict):
            _ParseMore(value)

def PrintMetadata(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile()
    i=s.open(input_file)
    if i: return i
    js=s.parse_buffer()
    if js is None: return -1

    if not "__metadata__" in js:
        print("file header does not contain a __metadata__ item",file=sys.stderr)
        return -2

    md=js["__metadata__"]
    if cmdLine['parse_more']:
        _ParseMore(md)    
    json.dump({"__metadata__":md},fp=sys.stdout,ensure_ascii=False,separators=(',',':'),indent=1)
    return 0

def HeaderKeysToFrozenSet(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile()
    s.open(input_file,quiet=True)
    js=s.parse_buffer()
    if js is None: return -1

    nonscalar_keys:[str]=[]
    scalar_keys:[str]=[]

    for key in js:
        if key=='__metadata__': continue
        v=js[key]
        isScalar=False
        if isinstance(v,dict):
            if 'shape' in v:
                if 0==len(v['shape']):
                    isScalar=True
        if isScalar:
            scalar_keys.append(key)
        else:
            nonscalar_keys.append(key)                                
    nonscalar_keys.sort()
    scalar_keys.sort()

    def printkeylist(kl):
        firstKey=True
        for key in kl:
            if firstKey: firstKey=False
            else: print(",")
            print(f'"{key}"',end='')
        print()

    print("# known scalar LoRA keys in a frozenset()")
    print("lora_keys_scalar=frozenset([")
    printkeylist(scalar_keys)
    print("])")

    print()
    print("# known non-scalar LoRA keys in a frozenset()")
    print("lora_keys_nonscalar=frozenset([")
    printkeylist(nonscalar_keys)
    print("])")

    return 0


def ExtractHeader(cmdLine:dict,input_file:str,output_file:str)->int:
    if os.path.exists(output_file):
        if cmdLine['force_overwrite']==False:
            print(f'output file "{output_file}" already exists, use -f to force overwrite',file=sys.stderr)
            return -1

    s=SafeTensorsFile()
    i=s.open(input_file)
    if i: return i

    hdrbuf=s.hdrbuf
    s.close() #close it in case user wants to write back to input_file itself
    with open(output_file,"wb") as fo:
        wn=fo.write(hdrbuf)
        if wn!=len(hdrbuf):
            print(f"write output file failed, tried to write {len(hdrbuf)} bytes, only wrote {wn} bytes",file=sys.stderr)
            return -1
    print(f"header saved to file {output_file}")
    return 0

def CheckLoRA(cmdLine:dict,input_file:str)->int:
    s=SafeTensorsFile()
    i=s.open(input_file)
    if i: return i
    js=s.parse_buffer()
    if js is None: return -1

    import lora_keys

    not_lora_items:[str]=[]
    allkeys=lora_keys.lora_keys_scalar.union(lora_keys.lora_keys_nonscalar)
    lora_items:dict[str,int]=dict.fromkeys(allkeys,int(0))
    for key in js:
        if key in lora_items:
            lora_items[key]+=int(1)
        else:
            if "__metadata__"!=key:
                not_lora_items.append(key)            

    hasError=False
    if len(not_lora_items)!=0:
        print("unrecognized items:")
        for x in not_lora_items: print(" ",x)
        hasError=True
    
    for k,v in lora_items.items():
        if (v==1): continue
        if (v==0):
            print(f"{k}: missing")
        else:
            print(f"{k}: used {v} times")    
        hasError=True            

    if hasError: return 1

    print("looks like an OK LoRA file")
    return 0
