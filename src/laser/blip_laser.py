import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class BLIPLaser(AbstractLaser):

    def __init__(self):
        super(AbstractLaser, self).__init__()

    @staticmethod
    def convert_name(name):
      if name == "qkv_proj":
          converted_name = "self_attn.qkv.weight"
      elif name == "proj":
          converted_name = "self_attn.projection.weight"
      elif name == "fc_1":
          converted_name = "mlp.fc1.weight"
      elif name == "fc_2":
          converted_name = "mlp.fc2.weight"
      elif name == "k_proj_crossattention":
          converted_name = "crossattention.self.key.weight"
      elif name == "q_proj_crossattention":
          converted_name = "crossattention.self.query.weight"
      elif name == "v_proj_crossattention":
          converted_name = "crossattention.self.value.weight"
      elif name == "fc_out_crossattention":
          converted_name = "crossattention.output.dense.weight"
      elif name == "out_proj_crossattention":
          converted_name = "crossattention.output.dense.weight"
      elif name == "None":
          converted_name = "None"
      else:
          raise AssertionError(f"Unhandled name {name}")

      return converted_name


    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_name):

        # Check for layer number match
        # If must be either -1 meaning modify all layers, or must match the given layer number
        if lnum_to_modify != -1 and not name.startswith(f"vision_model.encoder.layers.{lnum_to_modify}.{converted_name}"):
            return False
        # Check if layer type needs to be modified.
        #      'all', 'mlp', 'attn', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out'
        # If all, then modify all
        # If mlp, then only MLP
        # If attn, then only attn
        # Otherwise, update a given layer type
        # print("lnum = ", lnum_to_modify, "lname to modify = ", lname_to_modify, "converted_names = ", converted_names)
        # if type(converted_name) == list:
        #     modify_flag = any([name.endswith(f"{converted_name}") for converted_name in converted_names])
        # elif type(converted_names) == str:
        #     modify_flag = name.endswith(f"{converted_names}")
        # else:
        #     raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

        # return modify_flag
        return True

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        # print("In get_edited_model, lnum = ", lnum, " lname to modify = ", lname)
        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit

        converted_name = BLIPLaser.convert_name(lname)
        num_update = 0

        # for name, param in model.named_parameters():
        #   print(f"Parameter name: {name}")
        #   # print(f"Parameter value: {param}")
        #   print(f"Requires gradient: {param.requires_grad}\n")

        for name, param in model.named_parameters():
            # print(name)
            modify_flag = BLIPLaser._modify_layer(name=name,
                                                    lnum_to_modify=lnum,
                                                    lname_to_modify=lname,
                                                    converted_name=converted_name)

            if modify_flag:
                if logger is not None:
                    logger.log(f"Updating Layer: {name}")
                print(f"Updating Layer: {name}")
            else:
                continue

            if intervention == 'dropout':
                # For the sparsity analysis
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1)

            elif intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)

            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            BLIPLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters Llama: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
