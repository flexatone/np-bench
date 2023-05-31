---
theme: default
# background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
transition: slide-left
aspectRatio: '16/9'
favicon: /favicon.ico
title: "Out-Performing NumPy is Hard: When and How to Try with Your Own C-Extensions "
---

# Out-Performing NumPy is Hard: When and How to Try with Your Own C-Extensions

### Christopher Ariza

#### CTO, Research Affiliates

<style>
h1 {font-size: 0.8em;}
</style>




---
---
# About Me

<Transform :scale="1.2">
<v-clicks>

CTO at Research Affiliates

Python programmer since 2000

PhD in music composition, professor of music technology

Python for algorithmic composition, computational musicology

Since 2012, builder of financial systems in Python

Creator of StaticFrame, an alternative DataFrame library
</v-clicks>
</Transform>




---
---
# Sample Code Slide
<Transform :scale="1.5">

Code examples <uim-rocket />

```python {all|5|6|all} {lines:true, startLine:5}
>>> for x in range(30):
      x += 2
      print(x)
```
</Transform>



---
---
# Bullets


<Transform :scale="1.5">
<v-clicks>

- Item 1
- Item 2
    ```python
    >>> code
    ```
- Item 3
- Item 4
</v-clicks>
</Transform>

<style>
ul li {list-style-type: disc;}
</style>



---
layout: none
---
# Performance Results

<div class="absolute top-80px">
<img src="/first_true_1d.png" />
</div>

<style>
h1 {font-size: 2em; margin-top: 10px; margin-left: 20px;}
div {background-color: #666666;}
</style>

