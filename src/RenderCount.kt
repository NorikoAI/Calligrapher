package org.sourcekey.NorikoAI.Calligraphe

import kotlinx.html.js.onClickFunction
import react.*
import react.dom.button

val renderCount = functionalComponent<RProps> {
    val (count, setCount) = useState(0)
    useEffect {
        console.log("useEffect $count")
    }
    +"count $count"
    button {
        attrs {
            onClickFunction = {
                setCount(count + 1)
            }
        }
    }
}

fun RBuilder.renderCount() {
    child(renderCount)
}