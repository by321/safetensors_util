### Features

This is a light-weight utility program for [safetensors files](https://github.com/huggingface/safetensors "safetensors files") written in Python only (no major external dependencies). Currently it can do the following:


    Usage: safetensors_util.py [OPTIONS] COMMAND [ARGS]...

    Options:
      --version    Show the version and exit.
      -q, --quiet  quiet mode, don't print informational stuff
      --help       Show this message and exit.

    Commands:
      checklora    see if input file is a SD 1.x LoRA file
      extractdata  extract one tensor and save to file
      extracthdr   extract file header and save to output file
      header       print file header
      listkeys     print header key names (except \_\_metadata\_\_) as a Python list
      metadata     print only \_\_metadata\_\_ in file header
      writemd      read \_\_metadata\_\_ from json and write to safetensors file


The most useful thing is probably the read and write metadata commands. To read metadata:

        python safetensors_util.py metadata input_file.safetensors -pm

Many safetensors files, especially LoRA files, have a \_\_metadata\_\_ field in the file header that records training information, such as learning rates, number of epochs, number of images used, etc. You can see how your favorite file was trained and perhaps use some of the training parameters for your own model in the future.

The optional **-pm** flag is meant to make output more readable. Because safetensors files only allow string-to-string dictionary in header, non-string values need to be quoted. Basically the **-pm** flag tries to turn this:

        "ss_dataset_dirs":"{\"abc\": {\"n_repeats\": 2, \"img_count\": 60}}",

into this:

        "ss_dataset_dirs":{
         "abc":{
          "n_repeats":2,
          "img_count":60
         }
        },

You can create a JSON file containing a \_\_metadata\_\_ entry:

    {
         "__metadata__":{
              "Description": "Stable Diffusion 1.5 LoRA trained on cat pictures",
              "Trigger Words":["cat from hell","killer kitten"],
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
              "ss_network_dim":16
         }
    }

and write it to a safetensors file header using the **writemd** command:

        python safetensors_util.py writemd input.safetensors input.json output.safetensors
