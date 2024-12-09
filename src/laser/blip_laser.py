import torch
import os
from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class BLIPLaser(AbstractLaser):

    def __init__(self):
        super(AbstractLaser, self).__init__()

    @staticmethod
    def convert_name(name):
      if name == "k_proj":
          converted_name = "attention.self.key.weight"
      elif name == "q_proj":
          converted_name = "attention.self.query.weight"
      elif name == "v_proj":
          converted_name = "attention.self.value.weight"
      elif name == "out_proj":
          converted_name = "attention.output.dense.weight"
      elif name == "fc_in":
          converted_name = "intermediate.dense.weight"
      elif name == "fc_out":
          converted_name = "output.dense.weight"
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
        if lnum_to_modify != -1 and not name.startswith(f"text_decoder.bert.encoder.layer.{lnum_to_modify}.{converted_name}"):
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
            

            print("intervention = ", intervention)

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
            
            diff = torch.abs(mat_analysis_tensor - mat_analysis)
            frobenius_norm = torch.norm(diff, p='fro').item()
            
            # Compute the mean and max difference
            mean_diff = torch.mean(diff).item()
            max_diff = torch.max(diff).item()

            # Print the differences
            print(f"Frobenius Norm of the difference: {frobenius_norm:.6f}")
            print(f"Mean element-wise difference: {mean_diff:.6f}")
            print(f"Max element-wise difference: {max_diff:.6f}")

            BLIPLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        torch.save(model_edit.state_dict(), "reduced_model.pth")

        # Get the file sizes (in bytes)
        original_model_size = os.path.getsize("original_model.pth")
        reduced_model_size = os.path.getsize("reduced_model.pth")

        print(f"Original Model Size: {original_model_size / (1024 ** 2):.2f} MB")
        print(f"Reduced Model Size: {reduced_model_size / (1024 ** 2):.2f} MB")

        original_state_dict = model.state_dict()  # Get the state_dict of the original model
        reduced_state_dict = model_edit.state_dict()  # Get the state_dict of the reduced model

        differences = []

        # Iterate over the parameters in the original model and compare with the reduced model
        for name, original_param in original_state_dict.items():
            if name in reduced_state_dict:
                reduced_param = reduced_state_dict[name]
                
                # Check if the weights are the same
                if torch.allclose(original_param, reduced_param, atol=1e-8):  # tolerance of 1e-6
                    differences.append((name, "Same"))
                else:
                    differences.append((name, "Different"))
            else:
                differences.append((name, "Layer missing in reduced model"))

        # Print the results
        # for name, status in differences:
        #     print(f"Layer: {name} - {status}")
        

        assert num_update > 0, f"Must update some parameters Llama: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
