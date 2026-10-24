import ljg_vcoco_lib_test
import numpy as np
import torch
a = torch.tensor(torch.zeros(1,289),dtype=torch.bool)
print(1)

# data = np.zeros((289,3,256))
# for i in range(289):
#                     caption = vcoco_hoi_lib_2[i]
#                     text_encoding = self.blip(caption=caption, match_head='text_encoding')
#                     data[i] = text_encoding.cpu().numpy()
#                 savetxt('vcoco_lib.txt',data.reshape(-1,256),delimiter='\t')