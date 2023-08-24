import os, sys, json
import numpy

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
        self.hdrbuf=None    #header byet buffer
        self.header=None    #parsed header as a dict
        self.error=0

    def __del__(self):
        self.close_file()

    def close_file(self):
        if self.f is not None:
            self.f.close()
            self.f=None
            self.filename=""

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
    def open_file(filename:str,quiet=False):
        s=SafeTensorsFile()
        s.open(filename,quiet)
        return s

    def open(self,fn:str,quiet=False)->int:
        st=os.stat(fn)
        if st.st_size<8:
            raise SafeTensorsException.invalid_file(fn,"length less than 8 bytes")

        f=open(fn,"rb")
        b8=f.read(8) #read header size
        if len(b8)!=8:
            raise SafeTensorsException.invalid_file(fn,f"read only {len(b8)} bytes at start of file")
        headerlen=int.from_bytes(b8,'little',signed=False)
        if headerlen==0:
            raise SafeTensorsException.invalid_file(fn,"header size is 0")
        if (8+headerlen>st.st_size):
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
        return 0

    def get_header(self):
        if self.header is not None: return self.header
        if self.hdrbuf is None: raise Exception("SafetensorsFile no header buffer")
        self._CheckDuplicateHeaderKeys()
        self.header=json.loads(self.hdrbuf)

        return self.header

    def load_one_tensor(self,tensor_name:str):
        t=self.header[tensor_name]
        if t['dtype']=='F16':
            dt=numpy.half
        elif t['dtype']=='F32':
            dt=numpy.single
        #elif t['dtype']=='F64':
        #    dt=numpy.double
        else:
            msg=f"unsupported tensor data type in {self.filename}: {t['dtype']}"
            raise SafeTensorsException(msg)
        self.f.seek(8+self.headerlen+t['data_offsets'][0])
        bytesToRead=t['data_offsets'][1]-t['data_offsets'][0]
        bytes=self.f.read(bytesToRead)

        n=1
        for v in t['shape']: n=n*v
        vals=numpy.frombuffer(bytes,dtype=dt,count=n,offset=0)
        if dt!=numpy.single: vals=vals.astype(numpy.single)
        print(self.header[tensor_name],dt,n,bytesToRead)
        self.header[tensor_name]['values']=vals
        """a = np.array([1, 2, 3])
        print(a)
        print(a.dtype)
        # [1 2 3]
        # int64

        a_float = a.astype(np.float32)"""

    def load_data(self):
        self.get_header()
        #n:int=max([self.header[key].data_offsets[1] for key in self.header])
        #self.f.seek(8+self.headerlen)
        #databytes=self.f.read(n)
        #if len(databytes)!=n:
        #    msg=f"error reading file {self.filename}, tried to read {n} bytes, only read {len(databytes)}"
        #    raise SafeTensorsException(msg)
        d={}
        d['__metadata__']=''
        read_order:list[tuple(str,int)]=[]
        for k,v in self.header.items():
            if k=="__metadata__": #if metadata, just copy
                d['__metadata__']=v
                continue
            read_order.append((k,v['data_offsets'][0]))
        read_order.sort(key=lambda x:x[1])
        for x in read_order:
            self.load_one_tensor(x[0])
            #print("asdfas",x[0])
            #print(x[0],self.header[x[0]])
        # 'dtype': 'F16', 'shape': [32, 768], 'data_offsets': [491526, 540678]

    def copy_data_to_file(self,file_handle) -> int:

        self.f.seek(8+self.headerlen)
        bytesLeft:int=self.st.st_size - 8 - self.headerlen
        while bytesLeft>0:
            chunklen:int=min(bytesLeft,int(16*1024*1024)) #copy in blocks of 16 MB
            file_handle.write(self.f.read(chunklen))
            bytesLeft-=chunklen

        return 0
