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

### CTO, Research Affiliates

<style>
h1 {font-size: 1.5em;}
</style>




---
---
# About Me

<Transform :scale="1.5">
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
# Python Performance

<Transform :scale="1.5">
<v-clicks>

Python (using C PyObjects) is relatively slow

C-extensions offer opportunities for using C-types (at C performance)

With NumPy, we get flexible usage of C-arrays in Python

</v-clicks>
</Transform>


---
---
# My Journey

<Transform :scale="1.5">
<v-clicks>

Built StaticFrame leveraging NumPy

Performance studies identify opportunities

For pure Python, can implement routines in C-types

What about routines that are already using NumPy?
</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>
Some NumPy routines are implemented in Python

Some NumPy routines might do more than we need

</v-clicks>
</Transform>


---
---
# Can NumPy Routines be Optimized?

<Transform :scale="1.5">
<v-clicks>
Many NumPy routines are flexible

* Handle N-dimensional arrays

* Handle full diversity of dtypes

* Handle non-array (i.e., list, tuple) inputs

* May make defensive copies

More narrow routines might be able to out-perform flexible routines

</v-clicks>
</Transform>


---
---
# Case Study: Finding the First True in an Array

<Transform :scale="1.5">
<v-clicks>
Given a 1D Boolean array, what is the index of the first True

Given a 2D Boolean array, what are the indices of the first True per axis

Need to be able to search in both directions

Need to know if there are no True
</v-clicks>
</Transform>



---
layout: none
---
# Stack Overflow 1

<div class="absolute top-0px">
<img src="/screen-so-1.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>


---
layout: none
---
# Stack Overflow 2

<div class="absolute top-0px">
<img src="/screen-so-2.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>


---
layout: none
---
# NumPy Issue 2269

<div class="absolute top-0px">
<img src="/screen-gh-npy-issue-2269.png" style="height: 550px;" />
</div>

<style>
div {background-color: #666666;}
</style>




---
---
# Finding the First True: NumPy Options

<Transform :scale="1.5">

<pre>np.argmax()
</pre>

<pre>np.where()
</pre>

<pre>np.nonzero()
</pre>

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

