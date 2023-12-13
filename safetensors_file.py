import os, sys, json

class SafeTensorsException(Exception):
    def __init__(self, msg:str):
        self.msg=msg
        super().__init__(msg)

    @staticmethod
    def invalid_file(filename:str,whatiswrong:str):
        s=f"{filename} is not a valid .safetensors file: {whatiswrong}"
        return SafeTensorsException(msg=s)

    def __str__(self):
        return self.msg

class SafeTensorsChunk:
    def __init__(self,name:str,dtype:str,shape:list[int],offset0:int,offset1:int):
        self.name=name
        self.dtype=dtype
        self.shape=shape
        self.offset0=offset0
        self.offset1=offset1

class SafeTensorsFile:
    def __init__(self):
        self.f=None         #file handle
        self.hdrbuf=None    #header byte buffer
        self.header=None    #parsed header as a dict
        self.error=0

    def __del__(self):
        self.close_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()

    def close_file(self):
        if self.f is not None:
            self.f.close()
            self.f=None
            self.filename=""

    #test file: duplicate_keys_in_header.safetensors
    def _CheckDuplicateHeaderKeys(self):
        def parse_object_pairs(pairs):
            return [k for k,_ in pairs]

        keys=json.loads(self.hdrbuf,object_pairs_hook=parse_object_pairs)
        #print(keys)
        d={}
        for k in keys:
            if k in d: d[k]=d[k]+1
            else: d[k]=1
        hasError=False
        for k,v in d.items():
            if v>1:
                print(f"key {k} used {v} times in header",file=sys.stderr)
                hasError=True
        if hasError:
            raise SafeTensorsException.invalid_file(self.filename,"duplicate keys in header")

    @staticmethod
    def open_file(filename:str,quiet=False,parseHeader=True):
        s=SafeTensorsFile()
        s.open(filename,quiet,parseHeader)
        return s

    def open(self,fn:str,quiet=False,parseHeader=True)->int:
        st=os.stat(fn)
        if st.st_size<8: #test file: zero_len_file.safetensors
            raise SafeTensorsException.invalid_file(fn,"length less than 8 bytes")

        f=open(fn,"rb")
        b8=f.read(8) #read header size
        if len(b8)!=8:
            raise SafeTensorsException.invalid_file(fn,f"read only {len(b8)} bytes at start of file")
        headerlen=int.from_bytes(b8,'little',signed=False)

        if (8+headerlen>st.st_size): #test file: header_size_too_big.safetensors
            raise SafeTensorsException.invalid_file(fn,"header extends past end of file")

        if quiet==False:
            print(f"{fn}: length={st.st_size}, header length={headerlen}")
        hdrbuf=f.read(headerlen)
        if len(hdrbuf)!=headerlen:
            raise SafeTensorsException.invalid_file(fn,f"header size is {headerlen}, but read {len(hdrbuf)} bytes")
        self.filename=fn
        self.f=f
        self.st=st
        self.hdrbuf=hdrbuf
        self.error=0
        self.headerlen=headerlen
        if parseHeader==True:
            self._CheckDuplicateHeaderKeys()
            self.header=json.loads(self.hdrbuf)
        return 0

    def get_header(self):
        return self.header

    def load_one_tensor(self,tensor_name:str):
        self.get_header()
        if tensor_name not in self.header: return None

        t=self.header[tensor_name]
        self.f.seek(8+self.headerlen+t['data_offsets'][0])
        bytesToRead=t['data_offsets'][1]-t['data_offsets'][0]
        bytes=self.f.read(bytesToRead)
        if len(bytes)!=bytesToRead:
            print(f"{tensor_name}: length={bytesToRead}, only read {len(bytes)} bytes",file=sys.stderr)
        return bytes

    def copy_data_to_file(self,file_handle) -> int:

        self.f.seek(8+self.headerlen)
        bytesLeft:int=self.st.st_size - 8 - self.headerlen
        while bytesLeft>0:
            chunklen:int=min(bytesLeft,int(16*1024*1024)) #copy in blocks of 16 MB
            file_handle.write(self.f.read(chunklen))
            bytesLeft-=chunklen

        return 0
