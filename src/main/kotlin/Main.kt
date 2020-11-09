import kotlinext.js.requireAll
import kotlinx.browser.*
import kotlinx.css.*
import org.sourcekey.NorikoAI.Calligrapher.app
import react.dom.render
import styled.css
import styled.styledDiv


fun main() {
    requireAll(kotlinext.js.require.context("./", true, js("/\\.css$/")))
    render(document.getElementById("root")) {
        styledDiv {
            css {
                margin = "0"
                padding = "0"
                fontFamily = "sans-serif"
                textAlign = TextAlign.center
            }
            app()
        }
    }
}