def genotype(self):

    def _parse(weights, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      n = 4
      start = 0
      for i in range(self._steps):
        end = min(start + n, len(weights)) # 자꾸 W 빈배열 오류나서, 크기를 벗어나지 않도록 수정함
        #W = weights[start:end].copy()

        #디버깅용 출력 추가
        #print(f"Step {i}, n={n}, start={start}, end={end}, weights length={len(weights)}, W shape={W.shape}")

        W = weights[start:end].copy()

        # W가 비어있으면 출력하지 않도록 처리
        if W.size == 0:
          print(f"Step {i}, n={n}, start={start}, end={end}, weights length={len(weights)}")
          continue 

        #정상적으로 W가 비어 있지 않으면 shape 출력
        print(f"Step {i}, n={n}, start={start}, end={end}, weights length={len(weights)}, W shape={W.shape}")

        try:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:4]
        except ValueError: # This error happens when the 'none' op is not present in the ops
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:4]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if 'none' in PRIMITIVES[j]:
              if k != PRIMITIVES[j].index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            else:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[start+j][k_best], j, W[j][k_best]))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
