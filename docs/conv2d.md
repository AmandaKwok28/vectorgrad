Derivation of the conv2d dL/dX code

```py
# if we write it as the literal gradient
for cin in range(Cin):
    for p in range(H):
        for q in range(W):
            for cout in range(Cout):
                for i in range(Hout):
                    for j in range(Wout):
                        u = p - i
                        v = q - j
                        if 0 <= u < kh and 0 <= v < kw:
                            dX[cin, p, q] += out.grad[cout, i, j] * W[cout, cin, u, v]



# rewriting for efficiency we can be more idiomatic
# currentlyl we're trying to find the contribution of a pixel in X to all the channels in 
# the output Y. Asking: for each input, find the outputs that touched it
# instead, we can ask, for each output, add its contribution to input pixels it touched
for cout in range(Cout):
    for iout in range(Hout):
        for jout in range(Wout):
            g = out.grad[cout, iout, jout]   # the scalar value of that channel
            for cin in range(Cin):
                for u in range(kh):
                    for v in range(kw):
                        x = iout + u
                        y = jout + v
                        dX[cin, x, y] += g * W[cout, cin, u, v]  # we want sum of output channels



# now we can slice it
for cout in range(Cout):
    for cin in range(Cin):
        for iout in range(Hout):
            for jout in range(Wout):
                dX[cin, iout:iout+kh, jout:jout+kw] += out.grad[cout, iout, jout] * W[cout, cin]



```