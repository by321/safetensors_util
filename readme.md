### Features

This is a utility program for [safetensors files](https://github.com/huggingface/safetensors "safetensors files"). Currently it can do the following:

    
      Usage: safetensors_util.py [OPTIONS] COMMAND [ARGS]...
    
      Options:
        --version              Show the version and exit.
        -f, --force-overwrite  overwrite existing files
        -pm, --parse-more      when printing metadata, 'fix' unnecessarily escaped
                                            doublequotes to make text more readable
        --help                 Show this message and exit.
    
      Commands:
        extracthdr  extract file header and save to output file
        header        print file header
        metadata    print only __metadata__ in file header
    

The most interesting/useful thing is the metadata command:

        python safetensors_util.py metadata input_file.safetensors -pm

Many safetensors files, especially LoRA files, have a __metadata__ field in the file header that records training information, such as learning rates, number of epochs, number of images used, etc. You can see how your favorite file was trained and perhaps use some of the training parameters for your own model in the future.

The optional **-pm** flag is meant to make output more readable. Many LoRA files seem to contain unnecessarily escaped doublequote characters. Basically the **-pm** flag tries to turn this:

        "ss_dataset_dirs":"{\"abc\": {\"n_repeats\": 2, \"img_count\": 60}}",

into this:

        "ss_dataset_dirs":{
         "abc":{
          "n_repeats":2,
          "img_count":60
         }
        },

