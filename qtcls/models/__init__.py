# Copyright (c) QIU Tian. All rights reserved.

from .vit_example import *

__vars__ = vars()


def build_model(args):
    import torch
    from termcolor import cprint
    from .. import datasets
    from ..utils.dist import is_main_process
    from ..utils.io import checkpoint_loader

    model_lib = args.model_lib.lower()
    model_name = args.model.lower()

    if 'num_classes' in args.model_kwargs.keys():
        cprint(f"Warning: Do NOT set 'num_classes' in 'args.model_kwargs'. "
               f"Now fetching the 'num_classes' registered in 'qtcls/datasets/__init__.py'.", 'light_yellow')

    try:
        num_classes = datasets._num_classes[args.dataset.lower()]
    except KeyError:
        print(f"KeyError: 'num_classes' for the dataset '{args.dataset.lower()}' is not found. "
              f"Please register your dataset's 'num_classes' in 'qtcls/datasets/__init__.py'.")
        exit(1)

    args.model_kwargs['num_classes'] = num_classes

    pretrained = not args.no_pretrain and is_main_process()

    if model_lib == 'default':
        try:
            model = __vars__[model_name](**args.model_kwargs)
        except KeyError:
            print(f"KeyError: Model '{model_name}' is not found.")
            exit(1)

        if pretrained:
            found_specified_path = args.pretrain

            if found_specified_path:
                state_dict = torch.load(found_specified_path)
            else:
                raise FileNotFoundError(f"Pretrained model for '{model_name}' is not found. "
                                        f"Please register your pretrained path in 'qtcls/datasets/_pretrain_.py' "
                                        f"or set the argument '--no_pretrain'.")

            if 'model' in state_dict.keys():
                state_dict = state_dict['model']

            checkpoint_loader(model, state_dict, strict=False)

        return model

    if model_lib == 'timm':
        import timm

        if pretrained:
            found_specified_path = args.pretrain

            model = timm.create_model(model_name=model_name, pretrained=not found_specified_path,
                                      **args.model_kwargs)

            if found_specified_path:
                state_dict = torch.load(found_specified_path)
                if 'model' in state_dict.keys():
                    state_dict = state_dict['model']
                checkpoint_loader(model, state_dict, strict=False)

            return model

        model = timm.create_model(model_name=model_name, pretrained=False, **args.model_kwargs)
        return model

    raise ValueError(f"Model lib '{model_lib}' is not found.")
