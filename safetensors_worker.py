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

    def __del__(self):
        self.close()

    def close(self):
        if self.f is not None: self.f.close()
        self.f=None

    def open(self,fn:str,openMode='rb'):
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

            print(f"{fn}: length={st.st_size}, header length={headerlen}")
            hdrbuf=f.read(headerlen)
            if len(hdrbuf)!=headerlen:
                raise SafeTensorsException(fn,f"header size is {headerlen}, but read {len(hdrbuf)} bytes")
        except SafeTensorsException as e:
            print(e)
            return None

        self.fn=fn
        self.f=f
        self.st=st
        self.hdrbuf=hdrbuf
        return self.hdrbuf

def PrintHeader(cmdLine:dict,input_file:str) -> int:
    
    s=SafeTensorsFile()
    hdrbuf=s.open(input_file)
    if hdrbuf is None: return -1

    # All the .safetensors files I've seen have long key names, and as a result, 
    # the neither json nor pprint package prints text in very readable format,
    # so we print it ourselves, putting key name & value on one long line.
    # Note the print out is in Python format, not valid JSON format.
    js=json.loads(hdrbuf)

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
    
def ParseMore(d:dict):
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
            ParseMore(value)

def PrintMetadata(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile()
    hdrbuf=s.open(input_file)
    if hdrbuf is None: return -1

    js=json.loads(hdrbuf,parse_float=float,parse_int=int)
    if not "__metadata__" in js:
        print("file header does not contain a __metadata__ item",file=sys.stderr)
        return -2

    md=js["__metadata__"]
    if cmdLine['parse_more']:
        ParseMore(md)    
    json.dump({"__metadata__":md},fp=sys.stdout,ensure_ascii=False,separators=(',',':'),indent=1)
    #print(md)
    #MyPrettyPrint({"__metadata__":md})
    return 0

def ExtractHeader(cmdLine:dict,input_file:str,output_file:str)->int:
    if os.path.exists(output_file):
        if cmdLine['force_overwrite']==False:
            print(f'output file "{output_file}" already exists, use -f to force overwrite',file=sys.stderr)
            return -1

    s=SafeTensorsFile()
    hdrbuf=s.open(input_file)
    if hdrbuf is None: return -1

    s.close() #close it in case user wants to write back to input_file itself
    with open(output_file,"wb") as fo:
        wn=fo.write(hdrbuf)
        if wn!=len(hdrbuf):
            print(f"write output file failed, tried to write {len(hdrbuf)} bytes, only wrote {wn} bytes")
            return -1
    print(f"header saved to file {output_file}")
    return 0
