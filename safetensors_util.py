import os,sys,click

import safetensors_worker
# This file deals with command line only. If the command line is parsed successfully,
# we will call one of the functions in safetensors_worker.py.

readonly_input_file=click.argument("input_file", metavar='input_file',
                                   type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
force_overwrite_flag=click.option("-f","--force-overwrite",default=False,is_flag=True, show_default=True,
                                  help="overwrite existing files")
fix_ued_flag=click.option("-pm","--parse-more",default=False,is_flag=True, show_default=True,
                          help="when printing metadata, 'fix' unnecessarily escaped doublequotes to make text more readable" )

@click.group()

@click.version_option(version=2)

@click.pass_context
def cli(ctx):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

@cli.command(name="header",short_help="print file header")
@readonly_input_file
@click.pass_context
def cmd_header(ctx,input_file:str) -> int:
    sys.exit( safetensors_worker.PrintHeader(ctx.obj,input_file) )

@cli.command(name="metadata",short_help="print only __metadata__ in file header")
@readonly_input_file
@fix_ued_flag
@click.pass_context
def cmd_meta(ctx,input_file:str,parse_more:bool)->int:
    ctx.obj['parse_more'] = parse_more
    sys.exit( safetensors_worker.PrintMetadata(ctx.obj,input_file) )

@cli.command(name="listkeys",short_help="print header key names (except __metadata__) as a Python list")
@readonly_input_file
@click.pass_context
def cmd_keyspy(ctx,input_file:str) -> int:
    sys.exit( safetensors_worker.HeaderKeysToLists(ctx.obj,input_file) )

@cli.command(name="writemd",short_help="write metadata to safetensors file header")
@click.argument("in_st_file", metavar='input_st_file',
                 type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument("in_json_file", metavar='input_json_file',
                 type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument("out_st_file", type=click.Path(file_okay=True, dir_okay=False, writable=True),metavar='output_st_file')
@force_overwrite_flag
@click.pass_context
def cmd_writemd(ctx,in_st_file:str,in_json_file:str,out_st_file:str,force_overwrite:bool) -> int:
    """Read "__metadata__" from json file and write to safetensors header"""
    ctx.obj['force_overwrite'] = force_overwrite
    sys.exit( safetensors_worker.WriteMetadataToHeader(ctx.obj,in_st_file,in_json_file,out_st_file) )

@cli.command(name="extracthdr",short_help="extract file header and save to output file")
@readonly_input_file
@click.argument("output_file", type=click.Path(file_okay=True, dir_okay=False, writable=True),metavar='output_file')
@force_overwrite_flag
@click.pass_context
def cmd_extractheader(ctx,input_file:str,output_file:str,force_overwrite:bool) -> int:
    ctx.obj['force_overwrite'] = force_overwrite
    sys.exit( safetensors_worker.ExtractHeader(ctx.obj,input_file,output_file) )

@cli.command(name="checklora",short_help="see if input file is a LoRA file")
@readonly_input_file
@click.pass_context
def cmd_checklora(ctx,input_file:str)->int:
    sys.exit( safetensors_worker.CheckLoRA(ctx.obj,input_file) )

if __name__ == '__main__':
    cli(obj={})
