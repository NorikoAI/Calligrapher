package largeGrid

import kotlinx.html.DIV
import react.RBuilder
import react.RComponent
import react.RProps
import react.RState
import react.dom.RDOMBuilder
import react.dom.div
import ticker.Ticker

interface LargeGridProps : RProps {
    var classes: String?
}

class LargeGrid(props: LargeGridProps) : RComponent<LargeGridProps, RState>(props) {
    var thisRef: dynamic = null

    override fun RBuilder.render() {
        div(props.classes) {
            ref {
                ref { thisRef = it }
            }
        }
    }
}

fun RBuilder.largeGrid(classes: String? = null, block: RBuilder.() -> Unit) = child(LargeGrid::class) {
    attrs.classes = classes
    block(this)
}