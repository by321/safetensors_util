import json, sys
from safetensors_file import SafeTensorsFile
from safetensors_worker import _ParseMore

def get_object(tensorsfile: str) -> str:
	with SafeTensorsFile.open_file(tensorsfile, quiet=True) as s:
		js = s.get_header()

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
