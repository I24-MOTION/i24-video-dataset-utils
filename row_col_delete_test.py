import torch
import time

l = 8000
iterations = 500
idx = 599

t = torch.rand([l,l])


# test 1 - double list index
keep = [_ for _ in range(l)]
keep.remove(idx)

# total = 0
# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    
#     start = time.time()
#     t2 = t[keep,:][:,keep]
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     total += elapsed
# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))




# # test 2 - double tensor index
# keep = torch.tensor(keep)

# total = 0
# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    
#     start = time.time()
#     t2 = t[keep,:][:,keep]
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     total += elapsed
    
# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))


# # test 3 - concat_based approach

# total = 0
# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    
#     start = time.time()
    
#     t2 = torch.cat([torch.cat([t[:idx,:idx],t[idx+1:,:idx]],dim = 0),torch.cat([t[:idx,idx+1:],t[idx+1:,idx+1:]],dim = 0)],dim = 1)
    
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     total += elapsed
    
# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))



# # test 4 - shift and remove end approach
# total = 0

# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    
#     t2 = torch.clone(t)

#     start = time.time()   
#     t2 = torch.clone(t)

#     t2[idx:-1,:] = t[idx+1:,:]
#     t2[:,idx:-1] = t[:,idx+1:]
#     t2 = t2[:-1,:-1]
    
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     total += elapsed
    
# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))




# total = 0 
# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    
#     o = torch.ones(t.shape,dtype = int)

#     start = time.time()
#     o[:,idx] = 0
#     o[idx,:] = 0
#     o = o.view(-1).nonzero().squeeze(1)
#     t2 = t.view(-1)[o].view(l-1,l-1)
#     elapsed = time.time() - start
#     total += elapsed

# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))


# o = torch.ones(t.shape,dtype = bool)
# total = 0 
# for i in range(iterations):
#     if i%10 == 0:
#         print("On iteration {}".format(i))
    

#     start = time.time()
#     o[:,idx] = 0
#     o[idx,:] = 0
#     t2 = t[o].view(l-1,l-1)
#     elapsed = time.time() - start
#     total += elapsed

# print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))

total = 0 
for i in range(iterations):
    if i%10 == 0:
        print("On iteration {}".format(i))
    

    start = time.time()
    
    
    H,W = t.shape
    # Empty initialization, just a memory alloc
    t2 = torch.empty(H-1, W-1)
    # top-left block
    t2[:idx, :idx] = t[:idx, :idx]
    # top-right block
    t2[:idx, idx:] = t[:idx, idx+1:]
    # bottom-left block
    t2[idx:, :idx] = t[idx+1:, :idx]
    # bottom-right block
    t2[idx:, idx:] = t[idx+1:, idx+1:]
    
    elapsed = time.time() - start
    total += elapsed

print("Took {:.1f}s for {} iterations, {}s/it".format(total,iterations,total/iterations))




