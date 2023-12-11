import json, sys
from safetensors_file import SafeTensorsFile
from safetensors_worker import _ParseMore
"""
This script extracts the JSON header's ss_tag_frequency from a safetensors file, then outputs it.

TODO: gracefully error out on safetensors files without ["__metadata__"]["ss_tag_frequency"]
"""

def get_tags(tensorsfile: str) -> str:
	s = SafeTensorsFile.open_file(tensorsfile, quiet=True) # omit the first non-JSON line
	js = s.get_header()
	md = js["__metadata__"]
	_ParseMore(md) # pretty print the metadata
	stf = md["ss_tag_frequency"]
	return json.dumps(stf, ensure_ascii=False, separators=(', ', ': '), indent=4)

tensorsfile = sys.argv[1]
hdata = get_tags(tensorsfile)
print(hdata)
