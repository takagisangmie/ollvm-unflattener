# Android ARM64 支持修改总结

本文档总结了为支持Android ARM64架构对OLLVM去平坦化工具所做的修改。

## 主要修改内容

### 1. 架构支持 (`unflattener.py`)

#### 添加ARM64寄存器导入
```python
from miasm.arch.aarch64.regs import *
```

#### 修改符号执行中的寄存器处理
- 添加ARM64堆栈指针(SP)和帧指针(X29)的处理
- 在CALL/BL指令执行后恢复ARM64寄存器状态

### 2. 指令识别和处理

#### 条件指令支持
- 添加ARM64比较指令：`SUBS`, `CMP`
- 添加ARM64条件选择指令：`CSEL`
- 支持ARM64条件分支：`B.EQ`, `B.NE`, `B.cond`

#### 调用指令支持
- 支持ARM64分支链接指令：`BL`, `BLR`
- 替代x86的`CALL`指令

### 3. 二进制重写器 (`binrewrite.py`)

#### Keystone汇编器集成
```python
from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN
# ...
elif arch == 'aarch64l':
    BinaryRewriter.KS = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
```

#### ARM64指令长度处理
- ARM64分支指令固定长度：4字节
- x86跳转指令长度：6字节

#### 条件分支生成
- ARM64：生成`B.cond`指令
- x86：生成`Jcond`指令

### 4. 架构特定的控制流分析

#### 基本块识别
- 识别ARM64的`B`(无条件分支)指令
- 处理ARM64的条件分支模式

#### 函数调用跟踪
- 跟踪ARM64的`BL`/`BLR`指令
- 适当处理ARM64的链接寄存器

## 支持的ARM64指令

### 分支指令
- `B` - 无条件分支
- `B.EQ`, `B.NE`, `B.LT`, `B.GT` 等 - 条件分支
- `BL` - 分支并链接(函数调用)
- `BLR` - 分支并链接到寄存器

### 条件指令
- `CSEL` - 条件选择
- `CMP` - 比较
- `SUBS` - 减法并设置标志

### 数据移动
- `MOV` - 数据移动
- 各种加载/存储指令

## 使用方法

### 基本用法
```bash
python -m unflattener -i android_arm64_binary.so -o deobfuscated.so -t 0x12345678
```

### 跟踪所有调用
```bash
python -m unflattener -i android_arm64_binary.so -o deobfuscated.so -t 0x12345678 -a
```

## 测试验证

使用`test_arm64.py`脚本验证ARM64支持：

```bash
python test_arm64.py
```

该脚本验证：
1. ARM64依赖库是否正确导入
2. Keystone ARM64汇编器是否工作
3. BinaryRewriter ARM64初始化
4. 基本ARM64指令汇编

## 注意事项

### 架构检测
- Miasm自动检测二进制架构为`aarch64l`
- 工具根据架构自动选择相应的处理逻辑

### 兼容性
- 保持与现有x86/x64支持的完全兼容性
- 新增的ARM64支持不影响原有功能

### 限制
- 目前主要针对标准OLLVM控制流平坦化
- 可能需要根据具体的混淆模式进行调整

## 文件修改清单

1. `unflattener/unflattener.py` - 主要逻辑修改
2. `unflattener/binrewrite.py` - 二进制重写修改
3. `README.md` - 文档更新
4. `test_arm64.py` - 测试脚本(新增)
5. `samples/android/` - ARM64示例目录(新增)

## 后续开发建议

1. **更多ARM64指令支持**：根据实际遇到的混淆模式添加更多指令支持
2. **性能优化**：针对ARM64特点优化符号执行性能
3. **错误处理**：增强ARM64特有错误情况的处理
4. **测试用例**：添加更多ARM64二进制的测试用例
5. **文档完善**：补充ARM64特有的使用说明和最佳实践
