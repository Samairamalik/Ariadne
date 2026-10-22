import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import type { GraphNode, GraphEdge, RelationshipType, NodeType } from '../types'

interface Graph3DProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  selectedNodeId: string | null
  hoveredNodeId: string | null
  focusedNodeId: string | null
  onNodeClick: (node: GraphNode) => void
  onNodeHover: (node: GraphNode | null) => void
  activeRelationshipTypes: string[]
  activeNodeTypes: string[]
  selectedCluster: string
  searchQuery: string
  autoOrbit: boolean
  repulsion: number
}

// Ethereal Palette mapping for Node Types
export const NODE_TYPE_COLORS: Record<NodeType | string, string> = {
  concept: '#ff2d9b',    // Electric Magenta / Hot Pink
  model: '#00f0ff',      // Radiant Neon Cyan
  method: '#a855f7',     // Cyber Violet
  dataset: '#00ff88',    // Ethereal Emerald
  paper: '#ffb703',      // Solar Amber / Gold
  person: '#fb7185',     // Soft Coral
  other: '#94a3b8',      // Slate / Stardust
}

// Relationship line colors
export const RELATIONSHIP_COLORS: Record<RelationshipType | string, string> = {
  BUILDS_ON: 'rgba(0, 240, 255, 0.55)',
  SUPPORTS: 'rgba(0, 255, 136, 0.55)',
  CONTRADICTS: 'rgba(255, 23, 68, 0.85)',
  REQUIRED_BY: 'rgba(168, 85, 247, 0.55)',
  TANGENTIALLY_RELATED: 'rgba(148, 163, 184, 0.35)',
  IS_EXAMPLE_OF: 'rgba(255, 183, 3, 0.55)',
}

// Generate text billboard canvas for crisp node typography without clutter
function createLabelSprite(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return new THREE.Sprite()

  canvas.width = 512
  canvas.height = 128

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.font = '600 36px "Space Grotesk", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // Outer glow on text
  ctx.shadowColor = color
  ctx.shadowBlur = 12
  ctx.fillStyle = '#ffffff'
  ctx.fillText(text, 256, 64)

  const texture = new THREE.CanvasTexture(canvas)
  texture.minFilter = THREE.LinearFilter
  texture.needsUpdate = true

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
  })

  const sprite = new THREE.Sprite(material)
  sprite.scale.set(38, 9.5, 1)
  return sprite
}

export function Graph3D({
  nodes,
  edges,
  selectedNodeId,
  hoveredNodeId,
  focusedNodeId,
  onNodeClick,
  onNodeHover,
  activeRelationshipTypes,
  activeNodeTypes,
  selectedCluster,
  searchQuery,
  autoOrbit,
  repulsion,
}: Graph3DProps) {
  const fgRef = useRef<any>(null)
  const angleRef = useRef(0)
  const starsAddedRef = useRef(false)

  // Filter nodes & edges based on active filters & query
  const filteredData = useMemo(() => {
    const relSet = new Set(activeRelationshipTypes)
    const typeSet = new Set(activeNodeTypes)
    const query = searchQuery.trim().toLowerCase()

    const activeNodes = nodes.filter((n) => {
      if (typeSet.size > 0 && !typeSet.has(n.type)) return false
      if (selectedCluster !== 'all' && n.cluster !== selectedCluster) return false
      return true
    })

    const nodeIds = new Set(activeNodes.map((n) => n.id))

    const activeEdges = edges.filter((e) => {
      const srcId = typeof e.source === 'object' ? (e.source as any).id : e.source
      const tgtId = typeof e.target === 'object' ? (e.target as any).id : e.target
      if (!nodeIds.has(srcId) || !nodeIds.has(tgtId)) return false
      if (relSet.size > 0 && !relSet.has(e.type || e.predicate)) return false
      return true
    })

    return {
      nodes: activeNodes,
      links: activeEdges,
    }
  }, [nodes, edges, activeRelationshipTypes, activeNodeTypes, selectedCluster, searchQuery])

  // Neighbor lookup map for quick hover highlights
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

  // Configure forces & starfield when scene initializes
  useEffect(() => {
    if (!fgRef.current) return

    // Physics parameters
    fgRef.current.d3Force('charge')?.strength(-repulsion)
    fgRef.current.d3Force('link')?.distance(36)
    fgRef.current.d3Force('center')?.strength(0.85)

    // Add ethereal cosmic dust / starfield to background
    const scene = fgRef.current.scene?.()
    if (scene && !starsAddedRef.current) {
      starsAddedRef.current = true

      // Gentle directional and ambient lights for shiny specular highlights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
      scene.add(ambientLight)

      const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 1.2)
      dirLight1.position.set(200, 300, 400)
      scene.add(dirLight1)

      const dirLight2 = new THREE.DirectionalLight(0xff2d9b, 0.9)
      dirLight2.position.set(-300, -200, -300)
      scene.add(dirLight2)

      // Starfield particles
      const starCount = 600
      const starGeometry = new THREE.BufferGeometry()
      const starPositions = new Float32Array(starCount * 3)
      const starColors = new Float32Array(starCount * 3)

      for (let i = 0; i < starCount * 3; i += 3) {
        starPositions[i] = (Math.random() - 0.5) * 1600
        starPositions[i + 1] = (Math.random() - 0.5) * 1600
        starPositions[i + 2] = (Math.random() - 0.5) * 1600

        // Gentle cyan, violet, and silver stars
        const rChoice = Math.random()
        if (rChoice < 0.33) {
          starColors[i] = 0.0; starColors[i + 1] = 0.94; starColors[i + 2] = 1.0 // cyan
        } else if (rChoice < 0.66) {
          starColors[i] = 1.0; starColors[i + 1] = 0.18; starColors[i + 2] = 0.61 // pink
        } else {
          starColors[i] = 0.7; starColors[i + 1] = 0.7; starColors[i + 2] = 1.0 // starlight
        }
      }

      starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
      starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3))

      const starMaterial = new THREE.PointsMaterial({
        size: 2.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.45,
        blending: THREE.AdditiveBlending,
      })

      const starPoints = new THREE.Points(starGeometry, starMaterial)
      scene.add(starPoints)
    }
  }, [repulsion])

  // Camera flight when focusedNodeId changes
  useEffect(() => {
    if (!fgRef.current || !focusedNodeId) return

    const targetNode = filteredData.nodes.find((n) => n.id === focusedNodeId)
    if (!targetNode || targetNode.x == null) return

    const dist = 110
    const hyp = Math.hypot(targetNode.x, targetNode.y || 0, targetNode.z || 0) || 1
    const ratio = 1 + dist / hyp

    fgRef.current.cameraPosition(
      {
        x: targetNode.x * ratio,
        y: (targetNode.y || 0) * ratio + 15,
        z: (targetNode.z || 0) * ratio,
      },
      targetNode,
      1400
    )
  }, [focusedNodeId, filteredData.nodes])

  // Auto-orbit cinematic animation
  useEffect(() => {
    if (!autoOrbit) return
    let animId: number

    const orbit = () => {
      if (fgRef.current) {
        angleRef.current += 0.002
        const distance = 420
        fgRef.current.cameraPosition({
          x: distance * Math.sin(angleRef.current),
          z: distance * Math.cos(angleRef.current),
          y: 120 + 40 * Math.sin(angleRef.current * 0.5),
        })
      }
      animId = requestAnimationFrame(orbit)
    }

    animId = requestAnimationFrame(orbit)
    return () => cancelAnimationFrame(animId)
  }, [autoOrbit])

  return (
    <div className="absolute inset-0 w-full h-full z-0 overflow-hidden select-none">
      <ForceGraph3D
        ref={fgRef}
        graphData={filteredData}
        backgroundColor="#07060e"
        showNavInfo={false}
        enableNodeDrag={true}
        enableNavigationControls={true}
        warmupTicks={25}
        cooldownTicks={120}
        // Custom 3D Shiny Node rendering
        nodeThreeObject={(node: any) => {
          const isSelected = selectedNodeId === node.id
          const isHovered = hoveredNodeId === node.id
          const isNeighbor = hoveredNodeId ? neighborsMap.get(hoveredNodeId)?.has(node.id) : false
          const isQueryMatch = searchQuery
            ? node.name.toLowerCase().includes(searchQuery.toLowerCase())
            : true

          const baseColor = NODE_TYPE_COLORS[node.type] || '#ff2d9b'
          const group = new THREE.Group()

          // Determine opacity based on hover state
          const opacity = hoveredNodeId
            ? isHovered || isNeighbor
              ? 1
              : 0.15
            : isQueryMatch
            ? 0.95
            : 0.2

          const size = Math.max(6, Number(node.size || (node.degree || 1) * 1.8 + 6))

          // 1. Core Shiny Pearl / Sphere with high specular gloss
          const coreGeo = new THREE.SphereGeometry(size, 32, 32)
          const coreMat = new THREE.MeshPhongMaterial({
            color: baseColor,
            emissive: baseColor,
            emissiveIntensity: isSelected ? 0.9 : isHovered ? 0.8 : 0.45,
            specular: 0xffffff,
            shininess: 130,
            transparent: true,
            opacity: opacity,
          })
          const coreMesh = new THREE.Mesh(coreGeo, coreMat)
          group.add(coreMesh)

          // 2. Concentric Ethereal Outer Glow Aura
          const haloGeo = new THREE.SphereGeometry(size * 1.55, 24, 24)
          const haloMat = new THREE.MeshBasicMaterial({
            color: baseColor,
            transparent: true,
            opacity: (isHovered || isSelected ? 0.38 : 0.14) * opacity,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
          })
          const haloMesh = new THREE.Mesh(haloGeo, haloMat)
          group.add(haloMesh)

          // 3. Selection / Focus Pulse Ring
          if (isSelected || isHovered) {
            const ringGeo = new THREE.RingGeometry(size * 1.8, size * 2.05, 32)
            const ringMat = new THREE.MeshBasicMaterial({
              color: isSelected ? 0x00f0ff : baseColor,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.8,
              blending: THREE.AdditiveBlending,
            })
            const ringMesh = new THREE.Mesh(ringGeo, ringMat)
            ringMesh.rotation.x = Math.PI / 2
            group.add(ringMesh)
          }

          // 4. Ethereal Label Billboard
          const labelSprite = createLabelSprite(node.name || node.label, baseColor)
          labelSprite.position.set(0, size + 10, 0)
          labelSprite.material.opacity = (isHovered || isSelected ? 1 : 0.75) * opacity
          group.add(labelSprite)

          return group
        }}
        // Link styling & animated energy particles
        linkColor={(link: any) => {
          const type = link.type || link.predicate
          return RELATIONSHIP_COLORS[type] || 'rgba(0, 240, 255, 0.4)'
        }}
        linkWidth={(link: any) => {
          const type = link.type || link.predicate
          if (type === 'CONTRADICTS') return 2.6
          if (type === 'BUILDS_ON') return 1.8
          return 1.2
        }}
        linkOpacity={0.65}
        linkDirectionalParticles={(link: any) => {
          const type = link.type || link.predicate
          return type === 'CONTRADICTS' ? 4 : 2
        }}
        linkDirectionalParticleSpeed={(link: any) => {
          const type = link.type || link.predicate
          return type === 'CONTRADICTS' ? 0.009 : 0.005
        }}
        linkDirectionalParticleWidth={(link: any) => {
          const type = link.type || link.predicate
          return type === 'CONTRADICTS' ? 3.0 : 1.8
        }}
        linkDirectionalParticleColor={(link: any) => {
          const type = link.type || link.predicate
          return type === 'CONTRADICTS' ? '#ff1744' : '#00f0ff'
        }}
        onNodeClick={(node: any) => {
          onNodeClick(node)
        }}
        onNodeHover={(node: any) => {
          onNodeHover(node || null)
        }}
      />
    </div>
  )
}
