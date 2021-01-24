package org.sourcekey.NorikoAI.Calligrapher

import kotlinx.browser.window
import kotlin.random.Random

/**
 * 呢個Class嘅作用
 * 令ArryList有LinkList嘅型態
 */
open class ArrayLinkList<T> : ArrayList<T> {
    /**
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(vararg initElements: T) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        node = getOrNull(0)
    }

    /**
     * @param initNodeID 初始去指定Node,如冇set為第0個Node開始
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(initNodeID: Int, vararg initElements: T) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        if (0 <= initNodeID && initNodeID < initElements.size) {
            node = getOrNull(initNodeID)
        } else {
            node = getOrNull(0)
        }
    }

    /**
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(initElements: Array<out T>) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        node = getOrNull(0)
    }

    /**
     * @param initNodeID 初始去指定Node,如冇set為第0個Node開始
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(initNodeID: Int, initElements: Array<out T>) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        if (0 <= initNodeID && initNodeID < initElements.size) {
            node = getOrNull(initNodeID)
        } else {
            node = getOrNull(0)
        }
    }

    /**
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(initElements: ArrayList<T>) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        node = getOrNull(0)
    }

    /**
     * @param initNodeID 初始去指定Node,如冇set為第0個Node開始
     * @param initElements 初始化時一次過窒入所有元素
     * */
    constructor(initNodeID: Int, initElements: ArrayList<T>) : super() {
        for (initElement in initElements) {
            add(initElement)
        }
        if (0 <= initNodeID && initNodeID < initElements.size) {
            node = getOrNull(initNodeID)
        } else {
            node = getOrNull(0)
        }
    }


    interface OnElementsChangedListener {
        fun onElementsChanged()
    }

    private val onElementsChangedListeners: ArrayList<OnElementsChangedListener> = ArrayList()

    fun addOnElementsChangedListener(onElementsChangedListener: OnElementsChangedListener) {
        onElementsChangedListeners.add(onElementsChangedListener)
    }

    /**
     * 係米執行緊當Elements改變左時要執行嘅程序
     *
     * 用作防止自己Call自己 同 防超载
     * */
    private var isRunOnElementsChangedListeners = false

    /**
     * 執行當Elements改變左時要執行嘅程序
     *
     * 用作防止自己Call自己 同 防超载
     * 因啲Listener入面可能會Call呢個function
     * */
    private fun runOnElementsChangedListeners() {
        if (!isRunOnElementsChangedListeners) {
            isRunOnElementsChangedListeners = true
            window.setTimeout(fun() {
                try {
                    for (onElementsChangedListener in onElementsChangedListeners) {
                        onElementsChangedListener.onElementsChanged()
                    }
                } catch (e: dynamic) {
                    println(e)
                }
                isRunOnElementsChangedListeners = false
            }, 1000)
        }
    }

    var onAddListener: ((element: T)-> Unit)? = null

    override fun add(element: T): Boolean {
        val returnValue = super.add(element)
        runOnElementsChangedListeners()
        onAddListener?.invoke(element)
        return returnValue
    }

    override fun add(index: Int, element: T) {
        super.add(index, element)
        runOnElementsChangedListeners()
        onAddListener?.invoke(element)
    }

    override fun addAll(elements: Collection<T>): Boolean {
        val returnValue = super.addAll(elements)
        runOnElementsChangedListeners()
        elements.forEach { onAddListener?.invoke(it) }
        return returnValue
    }

    override fun addAll(index: Int, elements: Collection<T>): Boolean {
        val returnValue = super.addAll(index, elements)
        runOnElementsChangedListeners()
        elements.forEach { onAddListener?.invoke(it) }
        return returnValue
    }

    var onRemoveListener: ((element: T)-> Unit)? = null

    override fun clear() {
        super.clear()
        runOnElementsChangedListeners()
        this.forEach { onRemoveListener?.invoke(it) }
    }

    override fun remove(element: T): Boolean {
        val returnValue = super.remove(element)
        runOnElementsChangedListeners()
        onRemoveListener?.invoke(element)
        return returnValue
    }

    override fun removeAll(elements: Collection<T>): Boolean {
        val returnValue = super.removeAll(elements)
        runOnElementsChangedListeners()
        elements.forEach { onRemoveListener?.invoke(it) }
        return returnValue
    }

    override fun removeAt(index: Int): T {
        val returnValue = super.removeAt(index)
        runOnElementsChangedListeners()
        //.forEach { onRemoveListener?.invoke(it) }
        return returnValue
    }

    override fun removeRange(fromIndex: Int, toIndex: Int) {
        super.removeRange(fromIndex, toIndex)
        runOnElementsChangedListeners()
        //.forEach { onRemoveListener?.invoke(it) }
    }

    var onSetListener: ((element: T)-> Unit)? = null

    override fun set(index: Int, element: T): T {
        val returnValue = super.set(index, element)
        runOnElementsChangedListeners()
        onSetListener?.invoke(element)
        return returnValue
    }

    interface OnNodeEventListener<T> {
        fun onNodeChanged(preChangeNodeID: Int?, postChangeNodeID: Int?, preChangeNode: T?, postChangeNode: T?)
    }

    private val onNodeEventListeners: ArrayList<OnNodeEventListener<T>> = ArrayList()

    fun addOnNodeEventListener(onNodeEventListener: OnNodeEventListener<T>) {
        onNodeEventListeners.add(onNodeEventListener)
    }

    fun indexOfOrNull(element: T?): Int? {
        if (element == null) {
            return null
        } else {
            val index = super.indexOf(element)
            if (index == -1) {
                return null
            } else {
                return index
            }
        }
    }

    private var lastTimeNode: T? = null

    /**
     * 依家指住嘅Node
     */
    var node: T? = null
        get() {
            //校正, 防已刪Element return出去
            if (indexOfOrNull(field) == null) {
                field = null
            }
            //如List上有Element又未有指上任何Element就指住第0個
            if (field == null) {
                if (0 < size) {
                    field = getOrNull(Random.nextInt(0, size))
                }
            }

            return field
        }
        protected set(value) {
            //儲存低上次乜Node 用作畀lastTime()做返回上次嘅Node
            lastTimeNode = node
            //低改變前NodeID 作執行NodeEventListeners時供用
            val preChangeNodeID = nodeID
            //改變Node
            field = value
            //執行NodeEventListeners
            for (onNodeEventListener: OnNodeEventListener<T> in onNodeEventListeners) {
                onNodeEventListener.onNodeChanged(
                        preChangeNodeID, nodeID,
                        if (preChangeNodeID != null) {
                            getOrNull(preChangeNodeID)
                        } else {
                            null
                        }, node
                )
            }
        }

    /**
     * 家指住嘅Node嘅ID
     */
    val nodeID: Int?
        get() = indexOfOrNull(node)

    /**
     * 將個Node指住下一個Node
     */
    fun next(): Boolean {
        val index = indexOfOrNull(node)
        if (index != null) {
            node = getOrNull((index + 1) % size)
            return true
        }
        return false
    }

    /**
     * 將個Node指住上一個Node
     */
    fun previous(): Boolean {
        val index = indexOfOrNull(node)
        if (index != null) {
            node = getOrNull(if ((index - 1) == -1) {
                lastIndex
            } else {
                index - 1
            })
            return true
        }
        return false
    }

    /**
     * 將個Node指住指定嘅Node
     * @param nodeID 去指定嘅Node嘅ID
     */
    fun designated(nodeID: Int): Boolean {
        if (0 <= nodeID && nodeID < size) {
            node = getOrNull(nodeID)
            return true
        }
        return false
    }

    /**
     * 將個Node指住指定嘅Node
     * @param node 去指定嘅Node
     * @return true 響個List成功搵到並設置呢個node做現時嘅node
     * @return false 響個List搵唔到個node並不進行任何設置
     */
    fun designated(node: T): Boolean {
        val index = indexOfOrNull(node)
        return if (index != null) {
            designated(index)
        } else {
            false
        }
    }

    /**
     * 將個Node指住上次指住嘅Node
     */
    fun lastTime() {
        if (lastTimeNode != null) {
            node = lastTimeNode
        } else {
            node = node
        }
    }
}
