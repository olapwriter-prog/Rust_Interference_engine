// Minimal usage example (drop into an app module that depends on :rie).
package ai.rie.example

import ai.rie.RieEngine
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private var engine: RieEngine? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val out = TextView(this).also { setContentView(it) }
        lifecycleScope.launch(Dispatchers.Default) {
            // Push a model with: adb push model.gguf /sdcard/Android/data/<pkg>/files/
            val model = File(getExternalFilesDir(null), "model.gguf")
            val e = RieEngine.load(model.path).also { engine = it }
            withContext(Dispatchers.Main) { out.text = RieEngine.deviceInfo + "\n" + e.modelInfo + "\n\n" }
            val stats = e.chat(listOf(RieEngine.ChatMessage("user", "Explain big.LITTLE CPUs in one paragraph."))) { piece ->
                runOnUiThread { out.append(piece) }
                true
            }
            withContext(Dispatchers.Main) { out.append("\n\n$stats") }
        }
    }

    override fun onDestroy() {
        engine?.cancel()
        engine?.close()
        super.onDestroy()
    }
}
