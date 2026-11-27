;; Trecho pesado da prévia: y[t] += w0*x0[t] + w1*x1[t] + w2*x2[t] + w3*x3[t]
;; Ponteiros em bytes; SIMD de 4 em 4 amostras, com o resto uma a uma.
;; Depois de editar: node mobile/wasm/build.js
(module
  (import "env" "memory" (memory 1))
  (func (export "mac4")
    (param $y i32) (param $x0 i32) (param $x1 i32) (param $x2 i32) (param $x3 i32)
    (param $w0 f32) (param $w1 f32) (param $w2 f32) (param $w3 f32) (param $n i32)
    (local $i i32) (local $end i32)
    (local $v0 v128) (local $v1 v128) (local $v2 v128) (local $v3 v128)
    (local.set $v0 (f32x4.splat (local.get $w0)))
    (local.set $v1 (f32x4.splat (local.get $w1)))
    (local.set $v2 (f32x4.splat (local.get $w2)))
    (local.set $v3 (f32x4.splat (local.get $w3)))
    (local.set $end (i32.shl (i32.and (local.get $n) (i32.const -4)) (i32.const 2)))
    (block $simd_done
      (loop $simd
        (br_if $simd_done (i32.ge_s (local.get $i) (local.get $end)))
        (v128.store (i32.add (local.get $y) (local.get $i))
          (f32x4.add
            (v128.load (i32.add (local.get $y) (local.get $i)))
            (f32x4.add
              (f32x4.add
                (f32x4.mul (local.get $v0) (v128.load (i32.add (local.get $x0) (local.get $i))))
                (f32x4.mul (local.get $v1) (v128.load (i32.add (local.get $x1) (local.get $i)))))
              (f32x4.add
                (f32x4.mul (local.get $v2) (v128.load (i32.add (local.get $x2) (local.get $i))))
                (f32x4.mul (local.get $v3) (v128.load (i32.add (local.get $x3) (local.get $i))))))))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $simd)))
    (local.set $end (i32.shl (local.get $n) (i32.const 2)))
    (block $done
      (loop $tail
        (br_if $done (i32.ge_s (local.get $i) (local.get $end)))
        (f32.store (i32.add (local.get $y) (local.get $i))
          (f32.add
            (f32.load (i32.add (local.get $y) (local.get $i)))
            (f32.add
              (f32.add
                (f32.mul (local.get $w0) (f32.load (i32.add (local.get $x0) (local.get $i))))
                (f32.mul (local.get $w1) (f32.load (i32.add (local.get $x1) (local.get $i)))))
              (f32.add
                (f32.mul (local.get $w2) (f32.load (i32.add (local.get $x2) (local.get $i))))
                (f32.mul (local.get $w3) (f32.load (i32.add (local.get $x3) (local.get $i))))))))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $tail)))))
