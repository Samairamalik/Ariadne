import { useEffect, useMemo, useRef } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import type { GraphNode, GraphEdge, NodeType, SpatialFormation } from '../types'

interface Graph3DProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNodeId: string | null
  focusedNodeId: string | null
  onNodeClick: (node: GraphNode) => void
  onNodeRightClick?: (node: GraphNode) => void
  onNodeHover: (node: GraphNode | null) => void
  activeRelationshipCategories: string[]
  activeNodeTypes: string[]
  selectedCluster: string
  selectedDocId: string
  searchQuery: string
  autoOrbit: boolean
  onStopAutoOrbit?: () => void
  livingMotion: boolean
  formation: SpatialFormation
  repulsion: number
  maxNodes: number
  labelMode: 'smart' | 'all' | 'minimal'
  highlightedNodeNames: Set<string>
  highlightedEdgePairs: Set<string>
  egoNodeId: string | null
}

export const NODE_TYPE_COLORS: Record<NodeType | string, string> = {
  concept: '#e5b869',
  model: '#5ec2a8',
  method: '#e07a5f',
  dataset: '#95c687',
  paper: '#f3d078',
  person: '#d49a89',
  other: '#7ba7bc',
}

export type PredicateCategory =
  | 'ARCHITECTURE'
  | 'PERFORMANCE'
  | 'METHODOLOGY'
  | 'TENSION'
  | 'GENERAL'

export const PREDICATE_CATEGORY_META: Record<
  PredicateCategory,
  { label: string; color: string; hex: string }
> = {
  ARCHITECTURE: {
    label: 'Builds / Uses',
    color: 'rgba(229, 184, 105, 0.48)',
    hex: '#e5b869',
  },
  PERFORMANCE: {
    label: 'Supports / Achieves',
    color: 'rgba(94, 194, 168, 0.50)',
    hex: '#5ec2a8',
  },
  METHODOLOGY: {
    label: 'Proposes / Trains',
    color: 'rgba(123, 167, 188, 0.48)',
    hex: '#7ba7bc',
  },
  TENSION: {
    label: 'Contradicts / Replaces',
    color: 'rgba(217, 83, 79, 0.80)',
    hex: '#d9534f',
  },
  GENERAL: {
    label: 'Associative / Other',
    color: 'rgba(224, 122, 95, 0.36)',
    hex: '#e07a5f',
  },
}

export function categorizePredicate(pred: string | undefined): PredicateCategory {
  const p = (pred || '').toLowerCase().trim()
  if (
    p.includes('contradict') ||
    p.includes('replace') ||
    p.includes('differ') ||
    p.includes('challenge') ||
    p.includes('eliminat') ||
    p.includes('avoid') ||
    p.includes('vs') ||
    p.includes('dispens')
  ) {
    return 'TENSION'
  }
  if (
    p.includes('support') ||
    p.includes('achiev') ||
    p.includes('outperform') ||
    p.includes('improv') ||
    p.includes('boost') ||
    p.includes('show') ||
    p.includes('exceed') ||
    p.includes('evaluat') ||
    p.includes('compare')
  ) {
    return 'PERFORMANCE'
  }
  if (
    p.includes('build') ||
    p.includes('use') ||
    p.includes('consist') ||
    p.includes('contain') ||
    p.includes('include') ||
    p.includes('has') ||
    p.includes('have') ||
    p.includes('base') ||
    p.includes('adapt') ||
    p.includes('extend')
  ) {
    return 'ARCHITECTURE'
  }
  if (
    p.includes('requir') ||
    p.includes('propos') ||
    p.includes('present') ||
    p.includes('train') ||
    p.includes('comput') ||
    p.includes('predict') ||
    p.includes('defin') ||
    p.includes('formul') ||
    p.includes('model')
  ) {
    return 'METHODOLOGY'
  }
  return 'GENERAL'
}

export const RELATIONSHIP_COLORS: Record<string, string> = {
  BUILDS_ON: PREDICATE_CATEGORY_META.ARCHITECTURE.color,
  SUPPORTS: PREDICATE_CATEGORY_META.PERFORMANCE.color,
  CONTRADICTS: PREDICATE_CATEGORY_META.TENSION.color,
  REQUIRED_BY: PREDICATE_CATEGORY_META.METHODOLOGY.color,
  TANGENTIALLY_RELATED: PREDICATE_CATEGORY_META.GENERAL.color,
}

// Shared unit geometries so we never allocate hundreds of redundant sphere buffers
const UNIT_SPHERE_GEO = new THREE.SphereGeometry(1, 16, 16)
const UNIT_HALO_GEO = new THREE.SphereGeometry(1, 12, 12)
const UNIT_RING_GEO = new THREE.RingGeometry(1.55, 1.82, 28)

const labelTextureCache = new Map<string, THREE.CanvasTexture>()

function createLabelSprite(text: string, color: string, isHighlighted: boolean): THREE.Sprite {
  const cleanText = text.length > 28 ? text.slice(0, 26) + '…' : text
  const cacheKey = `${cleanText}__${color}__${isHighlighted ? '1' : '0'}`
  let texture = labelTextureCache.get(cacheKey)

  if (!texture) {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) return new THREE.Sprite()

    canvas.width = 512
    canvas.height = 112

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const textMetricsWidth = Math.min(480, Math.max(140, cleanText.length * 19 + 40))
    const pillX = (512 - textMetricsWidth) / 2
    ctx.fillStyle = isHighlighted ? 'rgba(229, 184, 105, 0.22)' : 'rgba(11, 14, 13, 0.78)'
    ctx.beginPath()
    ctx.roundRect(pillX, 22, textMetricsWidth, 68, 34)
    ctx.fill()

    if (isHighlighted) {
      ctx.strokeStyle = '#e5b869'
      ctx.lineWidth = 2.5
      ctx.stroke()
    }

    ctx.font = `${isHighlighted ? '700' : '600'} 28px "Space Grotesk", sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    ctx.shadowColor = isHighlighted ? '#e5b869' : color
    ctx.shadowBlur = isHighlighted ? 12 : 6
    ctx.fillStyle = isHighlighted ? '#fff9ec' : '#f3efe6'
    ctx.fillText(cleanText, 256, 56)

    texture = new THREE.CanvasTexture(canvas)
    texture.minFilter = THREE.LinearFilter
    texture.needsUpdate = true
    labelTextureCache.set(cacheKey, texture)
  }

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: isHighlighted ? 0.98 : 0.82,
    depthWrite: false,
  })

  const sprite = new THREE.Sprite(material)
  sprite.scale.set(isHighlighted ? 30 : 22, isHighlighted ? 6.6 : 4.8, 1)
  return sprite
}

export function Graph3D({
  nodes,
  edges,
  selectedNodeId,
  focusedNodeId,
  onNodeClick,
  onNodeRightClick,
  onNodeHover,
  activeRelationshipCategories,
  activeNodeTypes,
  selectedCluster,
  selectedDocId,
  searchQuery,
  autoOrbit,
  onStopAutoOrbit,
  livingMotion,
  formation,
  repulsion,
  maxNodes,
  labelMode,
  highlightedNodeNames,
  highlightedEdgePairs,
  egoNodeId,
}: Graph3DProps) {
  const fgRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const angleRef = useRef(0)
  const starsAddedRef = useRef(false)
  const initialCameraSetRef = useRef(false)
  const animatedRingsRef = useRef<THREE.Object3D[]>([])

  // Build full adjacency for Ego-network filtering
  const fullAdjacency = useMemo(() => {
    const map = new Map<string, Set<string>>()
    edges.forEach((e) => {
      const s = typeof e.source === 'object' ? (e.source as any).id : e.source
      const t = typeof e.target === 'object' ? (e.target as any).id : e.target
      if (!map.has(s)) map.set(s, new Set())
      if (!map.has(t)) map.set(t, new Set())
      map.get(s)!.add(t)
      map.get(t)!.add(s)
    })
    return map
  }, [edges])

  // Filter nodes & edges
  const filteredData = useMemo(() => {
    const catSet = new Set(activeRelationshipCategories)
    const typeSet = new Set(activeNodeTypes)

    let egoAllowed: Set<string> | null = null
    if (egoNodeId) {
      egoAllowed = new Set([egoNodeId])
      const hop1 = fullAdjacency.get(egoNodeId) ?? new Set()
      hop1.forEach((id) => {
        egoAllowed!.add(id)
        const hop2 = fullAdjacency.get(id)
        hop2?.forEach((id2) => egoAllowed!.add(id2))
      })
    }

    const baseNodes = nodes.filter((n) => {
      if (egoAllowed && !egoAllowed.has(n.id)) return false
      if (typeSet.size > 0 && !typeSet.has(n.type)) return false
      if (selectedCluster !== 'all' && n.cluster !== selectedCluster) return false
      if (selectedDocId !== 'all' && n.document_id !== selectedDocId) return false
      return true
    })

    const q = searchQuery.trim().toLowerCase()
    const prioritized = [...baseNodes].sort((a, b) => {
      const aHigh =
        a.id === selectedNodeId ||
        highlightedNodeNames.has(a.name.toLowerCase()) ||
        (q && a.name.toLowerCase().includes(q))
          ? 1
          : 0
      const bHigh =
        b.id === selectedNodeId ||
        highlightedNodeNames.has(b.name.toLowerCase()) ||
        (q && b.name.toLowerCase().includes(q))
          ? 1
          : 0
      if (aHigh !== bHigh) return bHigh - aHigh
      return (
        (b.degree || 0) * 2 +
        (b.mention_count || 0) -
        ((a.degree || 0) * 2 + (a.mention_count || 0))
      )
    })

    const activeNodes = prioritized.slice(0, maxNodes)
    const nodeIds = new Set(activeNodes.map((n) => n.id))

    const activeEdges = edges.filter((e) => {
      const srcId = typeof e.source === 'object' ? (e.source as any).id : e.source
      const tgtId = typeof e.target === 'object' ? (e.target as any).id : e.target
      if (!nodeIds.has(srcId) || !nodeIds.has(tgtId)) return false
      const cat = categorizePredicate(e.predicate || e.type)
      if (catSet.size > 0 && !catSet.has(cat)) return false
      return true
    })

    const total = activeNodes.length || 1
    const docList = Array.from(new Set(activeNodes.map((n) => n.document_id || 'unknown')))
    const typeOrder: NodeType[] = [
      'model',
      'paper',
      'method',
      'dataset',
      'concept',
      'person',
      'other',
    ]

    activeNodes.forEach((node, idx) => {
      if (formation === 'neural') {
        node.fx = undefined
        node.fy = undefined
        node.fz = undefined
      } else if (formation === 'galaxy') {
        const armCount = 4
        const armIndex =
          typeOrder.indexOf(node.type) >= 0
            ? typeOrder.indexOf(node.type) % armCount
            : idx % armCount
        const t = idx / total
        const radius = 50 + Math.pow(t, 0.78) * 440
        const spinAngle = t * Math.PI * 5.2 + (armIndex * (Math.PI * 2)) / armCount
        const spread = (1 - t * 0.5) * 34
        const deterministicJitter = ((idx * 137) % 100) / 100 - 0.5
        node.fx = Math.cos(spinAngle) * radius + deterministicJitter * spread
        node.fz = Math.sin(spinAngle) * radius + deterministicJitter * spread
        node.fy = Math.sin(t * Math.PI * 3 + armIndex) * 55 * (1 - t * 0.6)
      } else if (formation === 'sphere') {
        const shellMap: Record<string, number> = {
          paper: 130,
          model: 195,
          method: 260,
          dataset: 320,
          concept: 380,
          person: 410,
          other: 430,
        }
        const r = shellMap[node.type] || 330
        const phi = Math.acos(1 - (2 * (idx + 0.5)) / total)
        const theta = Math.PI * (1 + Math.sqrt(5)) * idx
        node.fx = r * Math.cos(theta) * Math.sin(phi)
        node.fy = r * Math.sin(theta) * Math.sin(phi)
        node.fz = r * Math.cos(phi)
      } else if (formation === 'pillars') {
        const docIdx = Math.max(0, docList.indexOf(node.document_id || 'unknown'))
        const numDocs = Math.max(1, docList.length)
        const pillarAngle = (docIdx / numDocs) * Math.PI * 2
        const ringRadius = numDocs > 1 ? 280 : 0
        const centerX = Math.cos(pillarAngle) * ringRadius
        const centerZ = Math.sin(pillarAngle) * ringRadius
        const helixAngle = idx * 0.55
        const localR = 38 + (node.degree > 5 ? 0 : 32)
        node.fx = centerX + Math.cos(helixAngle) * localR
        node.fz = centerZ + Math.sin(helixAngle) * localR
        node.fy = ((idx % 45) - 22) * 14
      }
    })

    return {
      nodes: activeNodes,
      links: activeEdges,
    }
  }, [
    nodes,
    edges,
    activeRelationshipCategories,
    activeNodeTypes,
    selectedCluster,
    selectedDocId,
    searchQuery,
    maxNodes,
    formation,
    selectedNodeId,
    highlightedNodeNames,
    egoNodeId,
    fullAdjacency,
  ])

  // Top 10 hubs get labels in smart mode so the view stays clean and readable
  const topLabelNodeIds = useMemo(() => {
    return new Set(filteredData.nodes.slice(0, 10).map((n) => n.id))
  }, [filteredData.nodes])

  const neighborsMap = useMemo(() => {
    const map = new Map<string, Set<string>>()
    filteredData.links.forEach((l) => {
      const s = typeof l.source === 'object' ? (l.source as any).id : l.source
      const t = typeof l.target === 'object' ? (l.target as any).id : l.target
      if (!map.has(s)) map.set(s, new Set())
      if (!map.has(t)) map.set(t, new Set())
      map.get(s)!.add(t)
      map.get(t)!.add(s)
    })
    return map
  }, [filteredData.links])

  const nodeNameById = useMemo(() => {
    const m = new Map<string, string>()
    nodes.forEach((n) => m.set(n.id, n.name.toLowerCase()))
    return m
  }, [nodes])

  // Set initial zoomed-out camera & configure forces and starfield
  useEffect(() => {
    if (!fgRef.current) return

    fgRef.current.d3Force('charge')?.strength(-repulsion)
    fgRef.current.d3Force('link')?.distance(formation === 'neural' ? 85 : 95)
    fgRef.current.d3Force('center')?.strength(0.5)
    if (formation !== 'neural') {
      fgRef.current.d3ReheatSimulation?.()
    }

    // Set a comfortable zoomed-out initial camera distance so the graph isn't in your face
    if (!initialCameraSetRef.current) {
      initialCameraSetRef.current = true
      setTimeout(() => {
        fgRef.current?.cameraPosition(
          { x: 0, y: 140, z: 780 },
          { x: 0, y: 0, z: 0 },
          900
        )
      }, 80)
    }

    // Allow zooming far out and close in smoothly
    const controls = fgRef.current.controls?.()
    if (controls) {
      controls.minDistance = 40
      controls.maxDistance = 2600
      controls.zoomSpeed = 1.2
      controls.rotateSpeed = 0.9
    }

    const scene = fgRef.current.scene?.()
    if (scene && !starsAddedRef.current) {
      starsAddedRef.current = true

      const ambientLight = new THREE.AmbientLight(0xf7f2e7, 0.9)
      scene.add(ambientLight)

      const dirLight1 = new THREE.DirectionalLight(0xe5b869, 1.35)
      dirLight1.position.set(250, 350, 400)
      scene.add(dirLight1)

      const dirLight2 = new THREE.DirectionalLight(0x5ec2a8, 1.05)
      dirLight2.position.set(-350, -250, -300)
      scene.add(dirLight2)

      const starCount = 700
      const starGeometry = new THREE.BufferGeometry()
      const starPositions = new Float32Array(starCount * 3)
      const starColors = new Float32Array(starCount * 3)

      for (let i = 0; i < starCount * 3; i += 3) {
        starPositions[i] = (Math.random() - 0.5) * 2800
        starPositions[i + 1] = (Math.random() - 0.5) * 2800
        starPositions[i + 2] = (Math.random() - 0.5) * 2800

        const rChoice = Math.random()
        if (rChoice < 0.45) {
          // Burnished Gold
          starColors[i] = 0.9
          starColors[i + 1] = 0.72
          starColors[i + 2] = 0.41
        } else if (rChoice < 0.8) {
          // Verdigris Jade
          starColors[i] = 0.37
          starColors[i + 1] = 0.76
          starColors[i + 2] = 0.66
        } else {
          // Warm Terracotta / Ivory
          starColors[i] = 0.88
          starColors[i + 1] = 0.48
          starColors[i + 2] = 0.37
        }
      }

      starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
      starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3))

      const starMaterial = new THREE.PointsMaterial({
        size: 2.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending,
      })

      const starPoints = new THREE.Points(starGeometry, starMaterial)
      starPoints.name = 'ariadne-starfield'
      scene.add(starPoints)
    }
  }, [repulsion, formation])

  // Stop auto-orbit immediately if user scrolls or drags so zoom/rotate is 100% reactive
  useEffect(() => {
    const el = containerRef.current
    if (!el || !autoOrbit) return
    const stop = () => onStopAutoOrbit?.()
    el.addEventListener('wheel', stop, { passive: true })
    el.addEventListener('pointerdown', stop, { passive: true })
    return () => {
      el.removeEventListener('wheel', stop)
      el.removeEventListener('pointerdown', stop)
    }
  }, [autoOrbit, onStopAutoOrbit])

  // Lightweight animation loop
  useEffect(() => {
    let frameId: number
    const startTime = performance.now()

    const animateLivingField = (now: number) => {
      const t = (now - startTime) * 0.001

      for (let i = 0; i < animatedRingsRef.current.length; i++) {
        const ring = animatedRingsRef.current[i]
        if (ring) {
          ring.rotation.z = t * 0.9 + i * 0.3
        }
      }

      if (livingMotion) {
        const scene = fgRef.current?.scene?.()
        const stars = scene?.getObjectByName?.('ariadne-starfield')
        if (stars) {
          stars.rotation.y = t * 0.01
        }
      }

      frameId = requestAnimationFrame(animateLivingField)
    }

    frameId = requestAnimationFrame(animateLivingField)
    return () => cancelAnimationFrame(frameId)
  }, [livingMotion])

  // Camera flight when focusedNodeId changes
  useEffect(() => {
    if (!fgRef.current || !focusedNodeId) return

    const targetNode = filteredData.nodes.find((n) => n.id === focusedNodeId)
    if (!targetNode || targetNode.x == null) return

    const dist = 160
    const hyp = Math.hypot(targetNode.x, targetNode.y || 0, targetNode.z || 0) || 1
    const ratio = 1 + dist / hyp

    fgRef.current.cameraPosition(
      {
        x: targetNode.x * ratio,
        y: (targetNode.y || 0) * ratio + 20,
        z: (targetNode.z || 0) * ratio,
      },
      targetNode,
      950
    )
  }, [focusedNodeId, filteredData.nodes])

  // Respect user's current zoom distance when autoOrbit is explicitly enabled
  useEffect(() => {
    if (!autoOrbit) return
    let animId: number

    const cam = fgRef.current?.camera?.()
    if (cam) {
      angleRef.current = Math.atan2(cam.position.x, cam.position.z)
    }

    const orbit = () => {
      if (fgRef.current) {
        const camera = fgRef.current.camera?.()
        const currentDist = camera
          ? Math.max(250, Math.hypot(camera.position.x, camera.position.z))
          : 760
        const currentY = camera ? camera.position.y : 140
        angleRef.current += 0.0018
        fgRef.current.cameraPosition({
          x: currentDist * Math.sin(angleRef.current),
          z: currentDist * Math.cos(angleRef.current),
          y: currentY,
        })
      }
      animId = requestAnimationFrame(orbit)
    }

    animId = requestAnimationFrame(orbit)
    return () => cancelAnimationFrame(animId)
  }, [autoOrbit])

  animatedRingsRef.current = []

  const hasActiveHighlights = highlightedNodeNames.size > 0
  const hasSearch = searchQuery.trim().length > 0

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full z-0 overflow-hidden select-none"
    >
      <ForceGraph3D
        ref={fgRef}
        graphData={filteredData}
        backgroundColor="rgba(0,0,0,0)"
        showNavInfo={false}
        enableNodeDrag={true}
        enableNavigationControls={true}
        warmupTicks={formation === 'neural' ? 45 : 20}
        cooldownTicks={90}
        d3AlphaMin={0.01}
        d3VelocityDecay={0.38}
        nodeThreeObject={(node: any) => {
          const isSelected = selectedNodeId === node.id
          const isNeighbor = selectedNodeId
            ? neighborsMap.get(selectedNodeId)?.has(node.id)
            : false
          const isRagOrPathHighlight = highlightedNodeNames.has(
            (node.name || '').toLowerCase()
          )
          const isQueryMatch = hasSearch
            ? (node.name || '').toLowerCase().includes(searchQuery.trim().toLowerCase()) ||
              (node.aliases || []).some((a: string) =>
                a.toLowerCase().includes(searchQuery.trim().toLowerCase())
              )
            : false

          const isSpotlighted = isSelected || isRagOrPathHighlight || isQueryMatch
          const isTopHub = topLabelNodeIds.has(node.id)

          const baseColor = isRagOrPathHighlight
            ? '#5ec2a8'
            : NODE_TYPE_COLORS[node.type] || '#e5b869'

          const group = new THREE.Group()

          let opacity = 0.92
          if (selectedNodeId && !hasActiveHighlights) {
            opacity = isSelected || isNeighbor ? 1.0 : 0.22
          } else if (hasActiveHighlights) {
            opacity = isRagOrPathHighlight || isSelected ? 1.0 : 0.16
          } else if (hasSearch) {
            opacity = isQueryMatch || isSelected ? 1.0 : 0.16
          }

          // Sleek, proportional node radius (not oversized blobs)
          const baseSize = Math.max(
            3.2,
            Math.min(10.5, Math.sqrt(node.degree || 1) * 1.85 + 2.8)
          )
          const size = isSpotlighted ? baseSize * 1.3 : baseSize

          // 1. Core Sphere (using shared geometry)
          const coreMat = new THREE.MeshPhongMaterial({
            color: baseColor,
            emissive: isSpotlighted ? '#fff5df' : baseColor,
            emissiveIntensity: isSelected ? 0.95 : isSpotlighted ? 0.75 : 0.42,
            specular: 0xf7e7c4,
            shininess: 110,
            transparent: true,
            opacity,
          })
          const coreMesh = new THREE.Mesh(UNIT_SPHERE_GEO, coreMat)
          coreMesh.scale.set(size, size, size)
          group.add(coreMesh)

          // 2. Subtle Glow Halo (only visible when opacity is high)
          if (opacity > 0.3) {
            const haloMat = new THREE.MeshBasicMaterial({
              color: isRagOrPathHighlight ? '#5ec2a8' : baseColor,
              transparent: true,
              opacity: (isSpotlighted ? 0.32 : 0.14) * opacity,
              blending: THREE.AdditiveBlending,
              depthWrite: false,
            })
            const haloMesh = new THREE.Mesh(UNIT_HALO_GEO, haloMat)
            const haloScale = size * (isSpotlighted ? 1.75 : 1.42)
            haloMesh.scale.set(haloScale, haloScale, haloScale)
            group.add(haloMesh)
          }

          // 3. Orbital Ring ONLY on selected or spotlighted nodes (prevents clutter)
          if (isSpotlighted) {
            const ringMat = new THREE.MeshBasicMaterial({
              color: isSelected ? 0xe5b869 : 0x5ec2a8,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.85,
              blending: THREE.AdditiveBlending,
              depthWrite: false,
            })
            const ringMesh = new THREE.Mesh(UNIT_RING_GEO, ringMat)
            ringMesh.scale.set(size, size, size)
            ringMesh.rotation.x = Math.PI / 2.3
            group.add(ringMesh)
            animatedRingsRef.current.push(ringMesh)
          }

          // 4. Clean Typography Label
          const showLabel =
            labelMode === 'all' ||
            (labelMode === 'smart' && (isSpotlighted || isNeighbor || isTopHub)) ||
            (labelMode === 'minimal' && isSpotlighted)

          if (showLabel && opacity > 0.25) {
            const labelSprite = createLabelSprite(
              node.name || node.label,
              NODE_TYPE_COLORS[node.type] || baseColor,
              isSpotlighted
            )
            labelSprite.position.set(0, size + 7, 0)
            labelSprite.material.opacity = (isSpotlighted ? 1 : 0.8) * opacity
            group.add(labelSprite)
          }

          return group
        }}
        linkCurvature={0.18}
        linkCurveRotation={(link: any) => {
          const idStr = String(link.id || '')
          return ((idStr.charCodeAt(0) || 0) % 6) * 0.5
        }}
        linkColor={(link: any) => {
          const sId = typeof link.source === 'object' ? link.source.id : link.source
          const tId = typeof link.target === 'object' ? link.target.id : link.target
          const sName = nodeNameById.get(sId) || ''
          const tName = nodeNameById.get(tId) || ''
          const isPathEdge =
            highlightedEdgePairs.has(`${sName}::${tName}`) ||
            highlightedEdgePairs.has(`${tName}::${sName}`)
          if (isPathEdge) return 'rgba(94, 194, 168, 0.95)'

          const cat = categorizePredicate(link.predicate || link.type)
          return PREDICATE_CATEGORY_META[cat]?.color || 'rgba(229, 184, 105, 0.35)'
        }}
        linkWidth={(link: any) => {
          const sId = typeof link.source === 'object' ? link.source.id : link.source
          const tId = typeof link.target === 'object' ? link.target.id : link.target
          const sName = nodeNameById.get(sId) || ''
          const tName = nodeNameById.get(tId) || ''
          const isPathEdge =
            highlightedEdgePairs.has(`${sName}::${tName}`) ||
            highlightedEdgePairs.has(`${tName}::${sName}`)
          if (isPathEdge) return 3.0
          if (selectedNodeId && (sId === selectedNodeId || tId === selectedNodeId)) return 2.0
          return 0.9
        }}
        linkOpacity={0.55}
        linkDirectionalParticles={(link: any) => {
          const sId = typeof link.source === 'object' ? link.source.id : link.source
          const tId = typeof link.target === 'object' ? link.target.id : link.target
          const sName = nodeNameById.get(sId) || ''
          const tName = nodeNameById.get(tId) || ''
          const isPathEdge =
            highlightedEdgePairs.has(`${sName}::${tName}`) ||
            highlightedEdgePairs.has(`${tName}::${sName}`)
          if (isPathEdge) return 4
          if (selectedNodeId && (sId === selectedNodeId || tId === selectedNodeId)) return 2
          return 0
        }}
        linkDirectionalParticleSpeed={0.008}
        linkDirectionalParticleWidth={2.8}
        linkDirectionalParticleColor={() => '#e5b869'}
        onNodeClick={(node: any) => {
          onNodeClick(node)
        }}
        onNodeRightClick={(node: any) => {
          onNodeRightClick?.(node)
        }}
        onNodeHover={(node: any) => {
          onNodeHover(node || null)
        }}
      />
    </div>
  )
}
