<script setup>
import { ref } from 'vue'
import axios from 'axios';

defineProps({
  
})

const inputText = ref('');
const outputText = ref('');

// 发送数据到后端
const sendToBackend = async () => {
  outputText.value = ''
  if (!inputText.value.trim()) {
    outputText.value = '请输入内容后再发送！';
    return;
  }

  try {
    // 发送请求（通过代理解决跨域，实际请求地址是 http://localhost:5001/api/process）
    const response = await axios.post('/api/process', {
      content: inputText.value
    });

    // 接收后端处理结果
    outputText.value = response.data.processed_content;
  } catch (error) {
    console.error('通信错误:', error);
    outputText.value = '后端未启动或通信失败，请检查后端服务！';
  }
};
</script>

<template>
 <div class="text-processor">
    <!-- 文本输入区域 -->
    <div class="input-section">
      <textarea 
        v-model="inputText" 
        placeholder="请输入文本，点击发送后..."
        rows="5"
      ></textarea>
      <button @click="sendToBackend">提交</button>
    </div>
    
    <!-- 结果显示区域 -->
    <div class="output-section">
      <h3>检测结果：</h3>
      <div class="result-box">
        {{ outputText || '等待数据...' }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.text-processor {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.input-section {
  margin-bottom: 30px;
}

textarea {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  resize: vertical;
  font-size: 16px;
}

button {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

button:hover {
  background-color: #2980b9;
}

.output-section {
  padding: 20px;
  border: 1px solid #eee;
  border-radius: 6px;
}

.result-box {
  min-height: 100px;
  margin-top: 10px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
  white-space: pre-wrap;
}
</style>
