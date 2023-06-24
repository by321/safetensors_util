import os, sys, json
from safetensors_file import SafeTensorsFile

def _need_force_overwrite(output_file:str,cmdLine:dict) -> bool:
    if cmdLine["force_overwrite"]==False:
        if os.path.exists(output_file):
            print(f'output file "{output_file}" already exists, use -f flag to force overwrite',file=sys.stderr)
            return True
    return False

def WriteMetadataToHeader(cmdLine:dict,in_st_file:str,in_json_file:str,out_st_file:str) -> int:
    if _need_force_overwrite(out_st_file,cmdLine): return -1

    with open(in_json_file,"rt") as f:
        inmeta=json.load(f)
    if not "__metadata__" in inmeta:
        print(f"file {in_json_file} does not contain a top-level __metadata__ item",file=sys.stderr)
        #json.dump(inmeta,fp=sys.stdout,indent=2)
        return -2
    inmeta=inmeta["__metadata__"] #keep only metadata
    #json.dump(inmeta,fp=sys.stdout,indent=2)

    s=SafeTensorsFile.open_file(in_st_file)
    js=s.get_header()
    if js is None: return -1

    if inmeta==[]:
        js.pop("__metadata__",0)
        print("loaded __metadata__ is an empty list, output file will not contain __metadata__ in header")
    else:
        print("adding __metadata__ to header:")
        json.dump(inmeta,fp=sys.stdout,indent=2)
        if isinstance(inmeta,dict):
            for k in inmeta:
                inmeta[k]=str(inmeta[k])
        else:
            inmeta=str(inmeta)
        js["__metadata__"]=inmeta
        print()

    newhdrbuf=json.dumps(js,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    newhdrlen:int=int(len(newhdrbuf))
    pad:int=((newhdrlen+7)&(~7))-newhdrlen #pad to multiple of 8

    with open(out_st_file,"wb") as f:
        f.write(int(newhdrlen+pad).to_bytes(8,'little'))
        f.write(newhdrbuf)
        if pad>0: f.write(bytearray([32]*pad))
        i:int=s.copy_data_to_file(f)
    if i==0:
        print(f"file {out_st_file} saved successfully")
    else:
        print(f"error {i} occurred when writing to file {out_st_file}")
    return i

def PrintHeader(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile.open_file(input_file)
    js=s.get_header()
    if js is None: return -1


    # All the .safetensors files I've seen have long key names, and as a result,
    # neither json nor pprint package prints text in very readable format,
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
    s=SafeTensorsFile.open_file(input_file)
    js=s.get_header()
    if js is None: return -1

    if not "__metadata__" in js:
        print("file header does not contain a __metadata__ item",file=sys.stderr)
        return -2

    md=js["__metadata__"]
    if cmdLine['parse_more']:
        _ParseMore(md)
    json.dump({"__metadata__":md},fp=sys.stdout,ensure_ascii=False,separators=(',',':'),indent=1)
    return 0

def HeaderKeysToLists(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile.open_file(input_file,quiet=True)
    js=s.get_header()
    if js is None: return -1

    _lora_keys:list[tuple(str,bool)]=[] # use list to sort by name
    for key in js:
        if key=='__metadata__': continue
        v=js[key]
        isScalar=False
        if isinstance(v,dict):
            if 'shape' in v:
                if 0==len(v['shape']):
                    isScalar=True
        _lora_keys.append((key,isScalar))
    _lora_keys.sort(key=lambda x:x[0])

    def printkeylist(kl):
        firstKey=True
        for key in kl:
            if firstKey: firstKey=False
            else: print(",")
            print(key,end='')
        print()

    print("# use list to keep insertion order")
    print("_lora_keys:list[tuple[str,bool]]=[")
    printkeylist(_lora_keys)
    print("]")

    return 0


def ExtractHeader(cmdLine:dict,input_file:str,output_file:str)->int:
    if _need_force_overwrite(output_file,cmdLine): return -1

    s=SafeTensorsFile.open_file(input_file)
    if s.error!=0: return s.error

    hdrbuf=s.hdrbuf
    s.close_file() #close it in case user wants to write back to input_file itself
    with open(output_file,"wb") as fo:
        wn=fo.write(hdrbuf)
        if wn!=len(hdrbuf):
            print(f"write output file failed, tried to write {len(hdrbuf)} bytes, only wrote {wn} bytes",file=sys.stderr)
            return -1
    print(f"raw header saved to file {output_file}")
    return 0


def _CheckLoRA_internal(s:SafeTensorsFile)->int:
    js=s.get_header()
    if js is None: return -1

    import lora_keys
    set_scalar=set()
    set_nonscalar=set()
    for x in lora_keys._lora_keys:
        if x[1]==True: set_scalar.add(x[0])
        else: set_nonscalar.add(x[0])

    bad_unknowns:list[str]=[] # unrecognized keys
    bad_scalars:list[str]=[] #bad scalar
    bad_nonscalars:list[str]=[] #bad nonscalar
    for key in js:
        if key in set_nonscalar:
            if js[key]['shape']==[]: bad_nonscalars.append(key)
            set_nonscalar.remove(key)
        elif key in set_scalar:
            if js[key]['shape']!=[]: bad_scalars.append(key)
            set_scalar.remove(key)
        else:
            if "__metadata__"!=key:
                bad_unknowns.append(key)

    hasError=False
    if len(set_scalar)>0:
        print("missing scalar keys:")
        for x in set_scalar: print(" ",x)
        hasError=True
    if len(set_nonscalar)>0:
        print("missing nonscalar keys:")
        for x in set_nonscalar: print(" ",x)
        hasError=True

    if len(bad_unknowns)!=0:
        print("unrecognized items:")
        for x in bad_unknowns: print(" ",x)
        hasError=True

    if len(bad_scalars)!=0:
        print("keys expected to be scalar but are nonscalar:")
        for x in bad_scalars: print(" ",x)
        hasError=True

    if len(bad_nonscalars)!=0:
        print("keys expected to be nonscalar but are scalar:")
        for x in bad_nonscalars: print(" ",x)
        hasError=True

    return (1 if hasError else 0)

def CheckLoRA(cmdLine:dict,input_file:str)->int:
    s=SafeTensorsFile.open_file(input_file)
    i:int=_CheckLoRA_internal(s)
    if i==0: print("looks like an OK LoRA file")
    return 0
