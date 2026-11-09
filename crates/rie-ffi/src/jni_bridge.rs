//! JNI entry points for `ai.rie.RieEngine` (see `bindings/android`).

use crate::{device_info_string, load_engine, parse_role, rie_default_config, to_params, to_stats, RieEngine, RieSampling};
use jni::objects::{JClass, JObject, JObjectArray, JString, JValue};
use jni::sys::{jboolean, jdouble, jfloat, jint, jlong, jstring};
use jni::JNIEnv;
use rie_core::chat::Message;
use std::sync::atomic::Ordering;

fn throw(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

fn engine<'a>(h: jlong) -> Option<&'a mut RieEngine> {
    unsafe { (h as *mut RieEngine).as_mut() }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeLoad(
    mut env: JNIEnv,
    _cls: JClass,
    path: JString,
    n_threads: jint,
    n_ctx: jint,
    use_gpu: jboolean,
) -> jlong {
    let path: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => {
            throw(&mut env, "invalid path");
            return 0;
        }
    };
    let mut cfg = rie_default_config();
    cfg.n_threads = n_threads;
    if n_ctx > 0 {
        cfg.n_ctx = n_ctx;
    }
    cfg.use_gpu = use_gpu as i32;
    match load_engine(&path, &cfg) {
        Ok(e) => Box::into_raw(Box::new(e)) as jlong,
        Err(m) => {
            throw(&mut env, &m);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeFree(_env: JNIEnv, _cls: JClass, h: jlong) {
    if h != 0 {
        unsafe { drop(Box::from_raw(h as *mut RieEngine)) };
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeCancel(_env: JNIEnv, _cls: JClass, h: jlong) {
    if let Some(e) = engine(h) {
        e.cancel.store(true, Ordering::SeqCst);
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeReset(_env: JNIEnv, _cls: JClass, h: jlong) {
    if let Some(e) = engine(h) {
        e.engine.reset();
    }
}

/// Runs generation (raw prompt when `roles` is null, chat otherwise), streaming pieces to
/// `callback.onToken(String): Boolean`. Returns a JSON stats string.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ai_rie_RieEngine_nativeGenerate(
    mut env: JNIEnv,
    _cls: JClass,
    h: jlong,
    prompt: JString,
    roles: JObjectArray,
    contents: JObjectArray,
    max_tokens: jint,
    temperature: jfloat,
    top_k: jint,
    top_p: jfloat,
    min_p: jfloat,
    repeat_penalty: jfloat,
    seed: jlong,
    callback: JObject,
) -> jstring {
    let Some(e) = engine(h) else {
        throw(&mut env, "engine is closed");
        return std::ptr::null_mut();
    };
    let s = RieSampling { temperature, top_k, top_p, min_p, repeat_penalty, seed: seed as u64, max_tokens };
    let params = to_params(&s);
    e.cancel.store(false, Ordering::SeqCst);
    let cancel = e.cancel.clone();

    let mut msgs = Vec::new();
    let is_chat = !roles.is_null();
    if is_chat {
        let n = env.get_array_length(&roles).unwrap_or(0);
        for i in 0..n {
            let r = env.get_object_array_element(&roles, i).ok().map(JString::from);
            let c = env.get_object_array_element(&contents, i).ok().map(JString::from);
            let (Some(r), Some(c)) = (r, c) else { continue };
            let r: String = env.get_string(&r).map(Into::into).unwrap_or_default();
            let c: String = env.get_string(&c).map(Into::into).unwrap_or_default();
            msgs.push(Message::new(parse_role(&r), c));
        }
    }
    let prompt: String = if prompt.is_null() { String::new() } else { env.get_string(&prompt).map(Into::into).unwrap_or_default() };

    let mut on_text = |piece: &str| -> bool {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        if callback.is_null() {
            return true;
        }
        let Ok(js) = env.new_string(piece) else { return true };
        let r = env.call_method(&callback, "onToken", "(Ljava/lang/String;)Z", &[JValue::Object(&js)]);
        let _ = env.delete_local_ref(js);
        match r.and_then(|v| v.z()) {
            Ok(keep) => keep,
            Err(_) => false,
        }
    };
    let res = if is_chat { e.engine.chat(&msgs, &params, &mut on_text) } else { e.engine.generate(&prompt, &params, &mut on_text) };
    match res {
        Ok(st) => {
            let s = to_stats(&st);
            let json = format!(
                "{{\"prompt_tokens\":{},\"reused_tokens\":{},\"generated_tokens\":{},\"prompt_tps\":{:.3},\"gen_tps\":{:.3},\"stop_reason\":{}}}",
                s.prompt_tokens, s.reused_tokens, s.generated_tokens, s.prompt_tps, s.gen_tps, s.stop_reason
            );
            env.new_string(json).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
        }
        Err(m) => {
            if !env.exception_check().unwrap_or(false) {
                throw(&mut env, &m.to_string());
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeBench(mut env: JNIEnv, _cls: JClass, h: jlong, pp: jint, tg: jint) -> jstring {
    let Some(e) = engine(h) else { return std::ptr::null_mut() };
    match e.engine.bench(pp.max(1) as usize, tg.max(0) as usize) {
        Ok(r) => env
            .new_string(format!("{{\"pp_tps\":{:.3},\"tg_tps\":{:.3}}}", r.prompt_tps, r.gen_tps))
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        Err(m) => {
            throw(&mut env, &m.to_string());
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeDeviceInfo(env: JNIEnv, _cls: JClass) -> jstring {
    env.new_string(device_info_string()).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_ai_rie_RieEngine_nativeModelInfo(env: JNIEnv, _cls: JClass, h: jlong) -> jstring {
    let p = unsafe { crate::rie_model_info(h as *const RieEngine) };
    if p.is_null() {
        return std::ptr::null_mut();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { crate::rie_string_free(p) };
    env.new_string(s).map(|s| s.into_raw()).unwrap_or(std::ptr::null_mut())
}

#[allow(dead_code)]
fn _unused(_: jdouble) {}
