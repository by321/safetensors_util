### Features

This is a utility program for [safetensors files](https://github.com/huggingface/safetensors "safetensors files"). Currently it can do the following:

    
    Usage: safetensors_util.py [OPTIONS] COMMAND [ARGS]...
    
    Options:
      --version  Show the version and exit.
      --help     Show this message and exit.
    
    Commands:
      checklora   see if input file is a SD 1.x LoRA file
      extracthdr  extract file header and save to output file
      header      print file header
      listkeys    print header key names (except __metadata__) as a Python list
      metadata    print only __metadata__ in file header
      writemd     write metadata to safetensors file header


The most useful thing is probably the metadata command:

        python safetensors_util.py metadata input_file.safetensors -pm

Many safetensors files, especially LoRA files, have a __metadata__ field in the file header that records training information, such as learning rates, number of epochs, number of images used, etc. You can see how your favorite file was trained and perhaps use some of the training parameters for your own model in the future.

The optional **-pm** flag is meant to make output more readable. Because safetensors files only allow string-to-string dictionary in header, non-string values need to be quoted. Basically the **-pm** flag tries to turn this:

        "ss_dataset_dirs":"{\"abc\": {\"n_repeats\": 2, \"img_count\": 60}}",

into this:

        "ss_dataset_dirs":{
         "abc":{
          "n_repeats":2,
          "img_count":60
         }
        },
You can create a JSON file containing a __metadata__ entry:

    {
         "__metadata__":{
              "Description": "Stable Diffusion 1.5 LoRA trained on cat pictures",
              "Trigger Words":["cat from hell"],
              "Base Model": "Stable Diffusion 1.5",
              "Training Info": {
                    "trainer": "modified Kohya SS",
                    "resolution":[512,512],
                    "lr":1e-6,
                    "text_lr":1e-6,
                    "schedule": "linear",
                    "text_scheduler": "linear",
                    "clip_skip": 0,
                    "regularization_images": "none"
              },
              "ss_network_alpha":16,
              "ss_network_dim":16,
         }
    }

and write it to safetensors file header using the writemd command:

        python safetensors_util.py writemd input.safetensors input.json output.safetensors
