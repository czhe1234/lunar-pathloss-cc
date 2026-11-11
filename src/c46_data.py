from c43_data import Data as Original,gather,R
import numpy as np,torch,json
class Data(Original):
    def __init__(self,ids):
        super().__init__(ids)
        where={int(i):j for j,i in enumerate(self.ids)}
        self.base=torch.full(self.y.shape,float('nan'),dtype=torch.float32,device='cuda')
        for shard in [0,1]:
            d=R/'features/c46_base';meta=json.loads((d/f'fit_part{shard}.json').read_text())
            assert meta['input_only'] and not meta['labels_masks_read']
            a=np.load(d/f'fit_part{shard}.npy',mmap_mode='r')
            for k,i in enumerate(meta['indices']):
                if i in where:self.base[where[i]].copy_(torch.from_numpy(np.array(a[k])))
        for d in (R/'runs/c38_1024_continue/predictions').glob('train_dev_c34_fresh_shard*'):
            meta=json.loads((d/'inference.json').read_text());assert meta['no_target_or_validity_mask_read']
            a=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
            for k,i in enumerate(json.loads((d/'indices.json').read_text())['index_rows']):
                if i in where:self.base[where[i]].copy_(torch.from_numpy(np.stack([np.array(v[k]) for v in a])))
        assert torch.isfinite(self.base).all()
    def batch(self,indices):
        x,tx,fs,y,m=super().batch(indices)
        ii=torch.as_tensor(indices,dtype=torch.long,device='cuda');base=self.base[self.lookup[ii]]
        return torch.cat((x,(base-fs)/20),1).contiguous(memory_format=torch.channels_last),tx,base,y,m
