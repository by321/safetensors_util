import json, sys
from safetensors_file import SafeTensorsFile
from safetensors_worker import _ParseMore

"""
$ python3 safetensors_m.py ss_network_module /path/to/file/summer_dress.safetensors
"networks.lora"

$ python3 safetensors_m.py nonexistent_module /path/to/file/summer_dress.safetensors
Error: Metadata does not contain a `nonexistent_module` item, did you spell it right?

$ python3 safetensors_m.py ss_network_module /path/to/file/weird_file.safetensors
Error: File header does not contain a `__metadata__` item

$ python3 safetensors_m.py ss_tag_frequency /path/to/file/trina.safetensors
{
    "6_trina": {
        "trina": 26, 
        " black hair": 20, 
        " hands on hips": 1, 
        " looking at viewer": 7
    }
}

"""

def get_object(tensorsfile: str) -> str:
	s = SafeTensorsFile.open_file(tensorsfile, quiet=True)
	js = s.get_header()
	s.close_file()

	if "__metadata__" not in js:
		return "Error: File header does not contain a `__metadata__` item"
	md = js["__metadata__"]
	if md_object not in md:
		return f'Error: Metadata does not contain a `{md_object}` item, did you spell it right?'
	_ParseMore(md) # pretty print the metadata
	stf = md[md_object]
	return json.dumps(stf, ensure_ascii=False, separators=(', ', ': '), indent=4)

md_object = sys.argv[1]
tensorsfile = sys.argv[2]
hdata = get_object(tensorsfile)

print(hdata)
