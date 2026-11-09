// Android library module wrapping librie.so. Build the native libraries first:
//   ../../scripts/build-android.sh            (CPU)
//   ../../scripts/build-android.sh --gpu      (CPU + Vulkan)
// which copies them into src/main/jniLibs/<abi>/librie.so.
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.rie"
    compileSdk = 35
    defaultConfig {
        minSdk = 26 // Android 8.0; Vulkan 1.1 compute needs API 28+ for the GPU backend
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // Model files are large and already compressed; never compress them in the APK.
    androidResources { noCompress += "gguf" }
}
