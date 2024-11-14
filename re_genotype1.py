def genotype(self, max_edges=4, use_adaptive_threshold=True):
    def _parse(weights, normal=True):
        PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights[start:end].copy()

            # 적응형 임계값 설정
            if use_adaptive_threshold:
                threshold = np.median(W)  # ex) 중간값을 임계값으로 설정 (또는 np.mean(W))
            else:
                threshold = 0.1  # 고정된 임계값 

            selected_edges = []
            for j in range(i + 2):
                max_weight = max(W[j][k] for k in range(len(W[j])) if 'none' not in PRIMITIVES[j] or k != PRIMITIVES[j].index('none'))
                if max_weight > threshold:
                    selected_edges.append((max_weight, j))

            # 가중치 기준으로 정렬 후, 최대 max_edges 개수만큼 선택
            selected_edges = sorted(selected_edges, key=lambda x: -x[0])[:max_edges]

            for weight, j in selected_edges:
                k_best = None
                for k in range(len(W[j])):
                    if 'none' in PRIMITIVES[j] and k == PRIMITIVES[j].index('none'):
                        continue
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                if k_best is not None:
                    # 가중치를 포함하여 gene에 추가
                    gene.append((PRIMITIVES[start + j][k_best], j, weight))

            start = end
            n += 1
        return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

    concat = range(2 + self._steps - self._multiplier, self._steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype