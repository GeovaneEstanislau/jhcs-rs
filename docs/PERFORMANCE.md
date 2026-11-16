# 📊 JHCS — Análise Teórica e Modelagem de Desempenho

> **Documento Oficial de Engenharia — jhcs-rs**  
> **Status:** Ativo | **Modelagem Numérica**

---

## 1. Complexidade Computacional das Operações Canônicas

| Operação | Dimensões | Complexidade de Tempo | Acessos à Memória | Razão Aritmética (FLOPs/Byte) |
|---|---|:---:|:---:|:---:|
| **Multiplicação de Matrizes (`matmul`)** | $A(M, K) \times B(K, N)$ | $O(M \cdot K \cdot N)$ | $O(M \cdot K + K \cdot N + M \cdot N)$ | Alta ($\approx K$ FLOPs por elemento transferido) |
| **Adição de Vetores (`vector_add`)** | $V(N) + W(N)$ | $O(N)$ | $O(3N)$ | Baixa ($\approx 0.33$ FLOPs por float lido/escrito) |
| **Ativação ReLU (`relu`)** | $V(N)$ | $O(N)$ | $O(2N)$ | Baixa ($\approx 0.5$ FLOPs por float lido/escrito) |

### Conclusão de Engenharia sobre a Razão Aritmética
* Operações com **alta razão aritmética** (`matmul`, convoluções 2D/3D) são as únicas que justificam o custo de transferência para a GPU ($T_{\text{transfer}}$).
* Operações com **baixa razão aritmética** (`vector_add`, `relu`, escalas) são limitadas por largura de banda de memória (*memory-bound*). Transferi-las isoladamente para a GPU causa desaceleração líquida; elas devem permanecer na CPU/SIMD, ou ser encadeadas como nós fundidos (*kernel fusion*) dentro de um DAG na GPU para evitar trânsito no barramento.
